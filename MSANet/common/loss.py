# common/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        assert predict.shape == target.shape, "the size of predict and target must be equal."
        N = predict.shape[0]

        # 全精度
        predict = predict.float()
        target = target.float()

        pre = torch.sigmoid(predict).view(N, -1)
        tar = target.view(N, -1)

        # 如果发现是 0/255 mask，这里自动规一到 0/1
        max_val = tar.max()
        if max_val > 1.5:   # 大于 1 基本就是 0-255
            tar = tar / 255.0

        pre = pre.clamp(0.0, 1.0)
        tar = tar.clamp(0.0, 1.0)

        intersection = (pre * tar).sum(-1)
        union = (pre + tar).sum(-1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice = torch.nan_to_num(dice, nan=0.0, posinf=1.0, neginf=0.0)

        score = 1.0 - dice.mean()
        return score

