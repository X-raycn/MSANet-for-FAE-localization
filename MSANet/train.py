import sys
import os
import math
import time
import numpy as np
import logging
from collections import defaultdict
import pprint
import shutil
import argparse

import torch
import torch.nn as nn
import torch.optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from tqdm import tqdm
from tqdm.contrib import tzip, tenumerate
from sklearn import metrics

from config import config, update_config, get_cfg_defaults
from models.models import Generator
from data.dataset_CelebA import CelebA
from common.utils import AverageMeter
import common.utils as utils
from common.metrics import calculate_pixel_score, calculate_img_score_np
from common.loss import DiceLoss


# ======================= 分布式初始化 =======================

def init_distributed():
    """
    通过环境变量初始化 DDP：
    - RANK / WORLD_SIZE / LOCAL_RANK 由 torchrun 传入
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        # 单卡 fallback
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    return rank, world_size, gpu


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('-c', '--cfg',
                        help='experiment configure file name',
                        default='./experiment/config.yaml',
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def main():
    # 初始化分布式训练
    rank, world_size, gpu = init_distributed()

    args = parse_args()
    config = get_cfg_defaults()
    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # ======================= 模型 & DDP =======================
    net = Generator()
    net = net.to(gpu)

    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[gpu],
        find_unused_parameters=False
    )

    optim = torch.optim.Adam(
        net.parameters(),
        lr=config.TRAIN.LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY
    )

    # 使用新的 GradScaler API，scale 稍微保守一点
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = 2  # 梯度累积步数

    # ======================= 日志 / 模型目录 =======================
    if rank == 0:
        if config.TRAIN.RESUME is True:
            model_path = config.TRAIN.CONTINUE_PATH
            state_dicts = torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu')
            net.module.load_state_dict(state_dicts['net'])
            optim.load_state_dict(state_dicts['opt'])
            begin_epoch = state_dicts['epoch'] + 1
            best_loss = state_dicts['best_loss']
            best_p_f1 = state_dicts['best_p_f1']

            if not os.path.exists(os.path.join(model_path, 'model_best_f1.pt')):
                best_p_f1 = 0.0
        else:
            model_path = os.path.join(
                config.OUTPUT_DIR,
                config.MODEL.NAME + '--' + time.strftime("%Y.%m.%d--%H-%M-%S")
            )
            os.makedirs(model_path, exist_ok=True)
            os.makedirs(os.path.join(model_path, 'images'), exist_ok=True)
            shutil.copy2(args.cfg, os.path.join(model_path, 'config.yaml'))
            shutil.copy2('./models/models.py', os.path.join(model_path, 'models.py'))

            begin_epoch = config.TRAIN.BEGIN_EPOCH
            best_loss = float('inf')
            best_p_f1 = 0.
    else:
        # 非主进程也需要这些变量，结构保持一致
        begin_epoch = config.TRAIN.BEGIN_EPOCH
        best_loss = float('inf')
        best_p_f1 = 0.
        model_path = "./temp"  # 非主进程用不到真实路径，仅占位

    # 只在主进程设置日志
    if rank == 0:
        utils.setup_logger(
            'train',
            model_path,
            'train',
            level=logging.INFO,
            screen=True,
            tofile=True
        )
        logger_train = logging.getLogger('train')
        logger_train.info(pprint.pformat(args))
        logger_train.info(config)
        logger_train.info(net)

        writer = SummaryWriter(log_dir=os.path.join(model_path, 'tb-logs'))
    else:
        logger_train = None
        writer = None

    # ======================= Loss 函数 =======================
    bce_loss = nn.BCEWithLogitsLoss().to(gpu)
    mse_loss = nn.MSELoss().to(gpu)
    ce_loss = nn.CrossEntropyLoss().to(gpu)
    dice_loss = DiceLoss().to(gpu)

    # ======================= Dataset & Dataloader =======================
    transform_train = T.Compose([
        T.Resize(config.TRAIN.IMAGE_SIZE),
        T.ToTensor(),
    ])
    transform_val = T.Compose([
        T.Resize(config.TEST.IMAGE_SIZE),
        T.ToTensor()
    ])

    train_dataset = CelebA(
        config.DATASET.ROOT,
        config.DATASET.DATA_LIST,
        config.DATASET.ATTR_PATH,
        mode='train',
        transforms=transform_train,
        origin=True
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE // world_size,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=config.WORKERS,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_dataset = CelebA(
        config.DATASET.ROOT,
        config.DATASET.DATA_LIST,
        config.DATASET.ATTR_PATH,
        mode='val',
        transforms=transform_val,
        origin=True
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=config.TEST.BATCH_SIZE // world_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=config.WORKERS,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ======================= Train Loop =======================
    for epoch in range(begin_epoch, config.TRAIN.EPOCHES):
        train_sampler.set_epoch(epoch)  # 每个 epoch 重新 shuffle 数据
        net.train()
        train_loss_meter = defaultdict(AverageMeter)
        optim.zero_grad()

        # （可选）在第一个 epoch 打印一次 mask 范围，确认是不是 0/255
        if epoch == begin_epoch and rank == 0:
            imgs_dbg, masks_dbg, _ = next(iter(trainloader))
            print("DEBUG mask range:", masks_dbg.min().item(), masks_dbg.max().item())

        if rank == 0:
            data_iter = tqdm(trainloader, dynamic_ncols=True)
        else:
            data_iter = trainloader

        for i, data in enumerate(data_iter):
            imgs, masks, labels = data
            imgs = imgs.to(gpu, non_blocking=True)
            masks = masks.to(gpu, non_blocking=True)
            labels = labels.to(gpu, non_blocking=True)

            # ==== 前向 + Loss ====
            with torch.amp.autocast('cuda'):
                reveal_masks, pred, edge = net(imgs)

                # 主分割 loss：BCE + Dice
                g_loss_mask = bce_loss(reveal_masks, masks) + dice_loss(reveal_masks, masks)
                g_loss_pred = ce_loss(pred, labels)

                # === 边界损失 ===
                with torch.no_grad():
                    lap = torch.tensor(
                        [[1, 1, 1],
                         [1, -8, 1],
                         [1, 1, 1]],
                        device=masks.device,
                        dtype=masks.dtype
                    ).view(1, 1, 3, 3)
                    edge_gt = F.conv2d(masks, lap, padding=1).abs()
                    edge_gt = (edge_gt > 0.1).float()

                edge = torch.sigmoid(edge).clamp(0.0, 1.0)
                edge_gt = edge_gt.clamp(0.0, 1.0)
                g_loss_edge = bce_loss(edge, edge_gt)
                if torch.isnan(g_loss_edge):
                    g_loss_edge = torch.tensor(0.0, device=g_loss_mask.device)

                g_loss = g_loss_mask + config.LOSS.LAMBDA_PRED * g_loss_pred + 0.15 * g_loss_edge

            # ==== DDP-safe NaN 检查 ====
            with torch.no_grad():
                # 当前 rank 是否 NaN
                local_has_nan = torch.isnan(g_loss.detach())
                # 把 bool → float，方便 all_reduce
                nan_flag = torch.tensor(
                    1.0 if local_has_nan else 0.0,
                    device=gpu
                )
                # 汇总所有 rank 的 NaN 数量
                if dist.is_initialized():
                    dist.all_reduce(nan_flag, op=dist.ReduceOp.SUM)
                nan_count = nan_flag.item()

            if nan_count > 0:
                if rank == 0:
                    print(f"⚠️ NaN detected at epoch {epoch}, step {i}, "
                          f"nan_count_across_ranks={nan_count}, skip this batch on all ranks.")
                optim.zero_grad(set_to_none=True)
                continue

            # ==== 梯度缩放 + 梯度累积 + 裁剪 ====
            g_loss = g_loss / accumulation_steps
            scaler.scale(g_loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trainloader):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            train_loss_meter['loss'].update(g_loss.item() * accumulation_steps)
            train_loss_meter['loss_mask'].update(g_loss_mask.item())
            train_loss_meter['loss_pred'].update(g_loss_pred.item())

        # 只在主进程记录训练日志
        if rank == 0:
            for k, v in train_loss_meter.items():
                writer.add_scalar(f'train/{k}', v.avg, epoch)

            msg = f"Train epoch {epoch}: "
            for k, v in train_loss_meter.items():
                msg += f"{k}: {v.avg:.6f} |"
            msg += f"lr: {optim.param_groups[0]['lr']} |"
            logger_train.info(msg)

        # ======================= Validation =======================
        net.eval()
        val_loss_meter = defaultdict(AverageMeter)
        img_list, mask_list, pre_list = [], [], []
        img_pred_list, img_socre_list, img_label_list = [], [], []

        with torch.no_grad():
            for i, data in enumerate(valloader):
                imgs, masks, labels = data
                imgs = imgs.to(gpu, non_blocking=True)
                masks = masks.to(gpu, non_blocking=True)
                labels = labels.to(gpu, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    reveal_masks, pred, edge = net(imgs)
                    g_loss_mask = bce_loss(reveal_masks, masks)
                    g_loss_pred = ce_loss(pred, labels)
                    g_loss = g_loss_mask + config.LOSS.LAMBDA_PRED * g_loss_pred

                reveal_masks = torch.sigmoid(reveal_masks)

                # 只在主进程收集少量样本用于可视化
                if rank == 0 and i % (1000 // (config.TEST.BATCH_SIZE // world_size)) == 0:
                    img_list.append(imgs)
                    mask_list.append(masks)
                    pre_list.append(reveal_masks)

                # 每 5 个 batch 计算一次像素级指标
                if i % 5 == 0:
                    batch_size = config.TEST.BATCH_SIZE // world_size
                    p_f1 = torch.zeros(batch_size, device=gpu)
                    mIoU = torch.zeros(batch_size, device=gpu)
                    mcc = torch.zeros(batch_size, device=gpu)

                    for j in range(batch_size):
                        reveal_masks_ = (reveal_masks[j].squeeze(0) >= 0.5).float()
                        masks_ = masks[j].squeeze(0)
                        p_f1[j], _, _, mIoU[j], mcc[j] = calculate_pixel_score(
                            reveal_masks_.flatten(), masks_.flatten()
                        )
                    p_f1, mIoU, mcc = p_f1.mean(), mIoU.mean(), mcc.mean()

                    val_loss_meter['p_f1'].update(p_f1.item())
                    val_loss_meter['mIoU'].update(mIoU.item())
                    val_loss_meter['mcc'].update(mcc.item())

                val_loss_meter['loss'].update(g_loss.item())
                val_loss_meter['loss_mask'].update(g_loss_mask.item())
                val_loss_meter['loss_pred'].update(g_loss_pred.item())

                img_pred_list += torch.max(torch.softmax(pred, dim=1), dim=1)[1].tolist()
                img_socre_list += torch.softmax(pred, dim=1)[:, 1].tolist()
                img_label_list += labels.tolist()

        # ======================= 保存模型 / 日志（主进程） =======================
        if rank == 0:
            try:
                img_auc = metrics.roc_auc_score(img_label_list, img_socre_list)
            except Exception:
                img_auc = np.zeros(1)
            _, _, _, img_f1, _, _, _, _ = calculate_img_score_np(
                img_pred_list, img_label_list
            )
            val_loss_meter['img_f1'].update(img_f1.item())
            val_loss_meter['img_auc'].update(float(img_auc))

            # 保存一部分可视化结果
            if img_list:
                utils.save_images(
                    torch.concat(img_list, dim=0),
                    torch.concat(mask_list, dim=0),
                    torch.concat(pre_list, dim=0),
                    config.TEST.BATCH_SIZE // world_size,
                    epoch,
                    os.path.join(model_path, 'images'),
                    256
                )

            for k, v in val_loss_meter.items():
                writer.add_scalar(f'val/{k}', v.avg, epoch)

            msg = f"Val epoch {epoch}: "
            for k, v in val_loss_meter.items():
                msg += f"{k}: {v.avg:.6f} |"
            logger_train.info(msg)

            # 每个 epoch 都更新一次通用 model.pt
            torch.save(
                {
                    'net': net.module.state_dict(),
                    'opt': optim.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'best_p_f1': best_p_f1
                },
                os.path.join(model_path, 'model.pt')
            )

            # 记录 best loss
            if val_loss_meter['loss'].avg < best_loss:
                torch.save(
                    {
                        'net': net.module.state_dict(),
                        'opt': optim.state_dict(),
                        'epoch': epoch,
                        'best_loss': val_loss_meter['loss'].avg,
                        'best_p_f1': best_p_f1
                    },
                    os.path.join(model_path, 'model_best_loss.pt')
                )
                best_loss = val_loss_meter['loss'].avg
                logger_train.info('Save best loss checkpoint.')

            # 记录 best p_f1
            if val_loss_meter['p_f1'].avg > best_p_f1:
                torch.save(
                    {
                        'net': net.module.state_dict(),
                        'opt': optim.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'best_p_f1': val_loss_meter['p_f1'].avg
                    },
                    os.path.join(model_path, 'model_best_f1.pt')
                )
                best_p_f1 = val_loss_meter['p_f1'].avg
                logger_train.info('Save best f1 checkpoint.')

    if rank == 0:
        writer.close()


if __name__ == '__main__':
    main()
