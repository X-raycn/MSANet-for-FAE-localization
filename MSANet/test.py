import sys
import os
import time
import random
import numpy as np
import logging
from collections import defaultdict
import argparse

import torch
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm.contrib import tenumerate
from sklearn import metrics

from config import update_config, get_cfg_defaults
from models.models import Generator
from data.dataset_CelebA import CelebA
from common.utils import AverageMeter
import common.utils as utils
from common.metrics import calculate_pixel_score, calculate_img_score_np

# ===== 可复现实验：固定随机种子=====
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')
    parser.add_argument('-d', '--dir',
                        help='experiment dir',
                        default='./result/Detection/runs/ab_model_aot_dct_h_1_m_2x4_cbam--2025.12.20-trans--19-33-00',
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser.parse_args()


def _unwrap_outputs(outputs):
    # 保留原来的兼容：len==3 / len==2；额外兼容 len>3 / dict
    if isinstance(outputs, (list, tuple)):
        if len(outputs) >= 2:
            return outputs[0], outputs[1]
        elif len(outputs) == 1:
            return outputs[0], None
    if isinstance(outputs, dict):
        rm = outputs.get("reveal_masks", outputs.get("mask", None))
        pd = outputs.get("pred", outputs.get("cls", None))
        return rm, pd
    if torch.is_tensor(outputs):
        return outputs, None
    return None, None


def main(args):
    model_path = args.dir
    config = get_cfg_defaults()
    update_config(config, os.path.join(model_path, 'config.yaml'))

    # log
    utils.setup_logger('test', model_path, 'test', level=logging.INFO, screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    net = Generator().to(device)

    #彻底不调用 torchsummary（否则 hook 可能残留，forward 会炸）
    total_params = sum(p.numel() for p in net.parameters())
    logger_test.info(f"Model params: {total_params/1e6:.3f} M")

    optim = torch.optim.Adam(net.parameters(), lr=config.TRAIN.LR)

    # 加载模型（保持你原逻辑）
    utils.load_model(os.path.join(model_path, 'model_best_f1.pt'), net, optim)

    net = nn.DataParallel(net)

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    transform_test = T.Compose([
        T.Resize(config.TEST.IMAGE_SIZE),
        T.ToTensor()
    ])

    # DataLoader 固定随机性
    g = torch.Generator()
    g.manual_seed(SEED)

    def _seed_worker(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    testloader = DataLoader(
        CelebA(config.DATASET.ROOT, config.DATASET.DATA_LIST, config.DATASET.ATTR_PATH,
               mode='test', transforms=transform_test, origin=False),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        num_workers=config.WORKERS,
        worker_init_fn=_seed_worker,
        generator=g
    )

    with torch.no_grad():
        net.eval()

        test_meter = defaultdict(AverageMeter)
        img_list, mask_list, pre_list = [], [], []
        img_pred_list, img_socre_list, img_label_list = [], [], []

        for i, data in tenumerate(testloader, dynamic_ncols=True):
            imgs, masks, labels, _ = data
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)

            outputs = net(imgs)
            reveal_masks, pred = _unwrap_outputs(outputs)

            if reveal_masks is None or pred is None:
                raise RuntimeError("Model output format unexpected: reveal_masks/pred is None")

            g_loss_mask = bce_loss(reveal_masks, masks)
            g_loss_pred = ce_loss(pred, labels)
            g_loss = g_loss_mask + config.LOSS.LAMBDA_PRED * g_loss_pred

            reveal_masks = torch.sigmoid(reveal_masks)

            p_f1 = torch.zeros(config.TEST.BATCH_SIZE, device=device)
            precision = torch.zeros(config.TEST.BATCH_SIZE, device=device)
            recall = torch.zeros(config.TEST.BATCH_SIZE, device=device)
            mIoU = torch.zeros(config.TEST.BATCH_SIZE, device=device)
            mcc = torch.zeros(config.TEST.BATCH_SIZE, device=device)
            p_auc = np.zeros(config.TEST.BATCH_SIZE)

            for j in range(config.TEST.BATCH_SIZE):
                reveal_masks_ = reveal_masks[j].squeeze(0)
                masks_ = masks[j].squeeze(0)

                y_true = masks_.cpu().numpy().flatten()
                y_score = reveal_masks_.cpu().numpy().flatten()
                if len(np.unique(y_true)) < 2:
                    p_auc[j] = np.nan
                else:
                    p_auc[j] = metrics.roc_auc_score(y_true, y_score)

                reveal_masks_bin = (reveal_masks_ >= 0.5).float()
                p_f1[j], precision[j], recall[j], mIoU[j], mcc[j] = calculate_pixel_score(
                    reveal_masks_bin.flatten(), masks_.flatten()
                )

            test_meter['loss'].update(float(g_loss.item()))
            test_meter['loss_mask'].update(float(g_loss_mask.item()))
            test_meter['loss_pred'].update(float(g_loss_pred.item()))
            test_meter['p_f1'].update(float(p_f1.mean()))
            test_meter['precision'].update(float(precision.mean()))
            test_meter['recall'].update(float(recall.mean()))
            test_meter['mIoU'].update(float(mIoU.mean()))
            test_meter['mcc'].update(float(mcc.mean()))
            test_meter['p_auc'].update(float(np.nanmean(p_auc)))

            img_pred_list += torch.max(torch.softmax(pred, dim=1), dim=1)[1].tolist()
            img_socre_list += torch.softmax(pred, dim=1)[:, 1].tolist()
            img_label_list += labels.tolist()

            if i % (1500 // config.TEST.BATCH_SIZE) == 0:
                img_list.append(imgs)
                mask_list.append(masks)
                pre_list.append(reveal_masks)

        if len(np.unique(img_label_list)) < 2:
            img_auc = float('nan')
        else:
            img_auc = metrics.roc_auc_score(img_label_list, img_socre_list)

        _, _, _, img_f1, _, _, _, _ = calculate_img_score_np(img_pred_list, img_label_list)
        test_meter['img_f1'].update(float(img_f1))
        test_meter['img_auc'].update(float(img_auc))

        save_images(torch.concat(img_list, dim=0),
                    torch.concat(mask_list, dim=0),
                    torch.concat(pre_list, dim=0),
                    config.TEST.BATCH_SIZE,
                    model_path,
                    256)

        logger_test.info(f"Test result: "
                         f"loss: {test_meter['loss'].avg:.6f} |"
                         f"loss_mask: {test_meter['loss_mask'].avg:.6f} |"
                         f"loss_pred: {test_meter['loss_pred'].avg:.6f} |"
                         f"p_f1: {test_meter['p_f1'].avg:.6f} |"
                         f"precision: {test_meter['precision'].avg:.6f} |"
                         f"recall: {test_meter['recall'].avg:.6f} |"
                         f"mIoU: {test_meter['mIoU'].avg:.6f} |"
                         f"mcc: {test_meter['mcc'].avg:.6f} |"
                         f"p_auc: {test_meter['p_auc'].avg:.6f} |"
                         f"img_f1: {test_meter['img_f1'].avg:.6f} |"
                         f"img_auc: {test_meter['img_auc'].avg:.6f} |"
                         )


def save_images(images, masks, reveal_masks, rows, folder, resize_to=None):
    masks = masks.expand_as(images)
    reveal_masks = reveal_masks.expand_as(images)
    images_list = torch.split(images, rows, dim=0)
    masks_list = torch.split(masks, rows, dim=0)
    reveal_masks_list = torch.split(reveal_masks, rows, dim=0)
    stack_images = []
    for data in zip(images_list, masks_list, reveal_masks_list):
        stack_images.append(torch.concat(data, dim=0))
    stack_images = torch.concat(stack_images, dim=0)

    if resize_to is not None:
        stack_images = F.interpolate(stack_images, size=resize_to)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f'test_{timestamp}.png'
    save_path = os.path.join(folder, filename)

    torchvision.utils.save_image(stack_images, save_path, nrow=rows, normalize=False)
    print(f"保存结果到 {save_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
