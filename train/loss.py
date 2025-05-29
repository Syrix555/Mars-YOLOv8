import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.bbox import bboxDecode, iou, bbox2dist
from train.tal import TaskAlignedAssigner


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iouv = iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iouv) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class DetectionLoss(object):
    def __init__(self, mcfg, model):
        self.model = model
        self.mcfg= mcfg
        self.layerStrides = model.layerStrides
        self.assigner = TaskAlignedAssigner(topk=self.mcfg.talTopk, num_classes=self.mcfg.nc, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bboxLoss = BboxLoss(self.mcfg.regMax).to(self.mcfg.device)

    def preprocess(self, targets, batchSize, scaleTensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batchSize, 0, ne - 1, device=self.mcfg.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batchSize, counts.max(), ne - 1, device=self.mcfg.device)
            for j in range(batchSize):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = out[..., 1:5].mul_(scaleTensor)
        return out

    def __call__(self, preds, targets):
        """
        preds shape:
            preds[0]: (B, regMax * 4 + nc, 80, 80)
            preds[1]: (B, regMax * 4 + nc, 40, 40)
            preds[2]: (B, regMax * 4 + nc, 20, 20)
        targets shape:
            (?, 6)
        """
        loss = torch.zeros(3, device=self.mcfg.device)  # box, cls, dfl

        batchSize = preds[0].shape[0]
        no = self.mcfg.nc + self.mcfg.regMax * 4

        # predictioin preprocess
        predBoxDistribution, predClassScores = torch.cat([xi.view(batchSize, no, -1) for xi in preds], 2).split((self.mcfg.regMax * 4, self.mcfg.nc), 1)
        predBoxDistribution = predBoxDistribution.permute(0, 2, 1).contiguous() # (batchSize, 80 * 80 + 40 * 40 + 20 * 20, regMax * 4)
        predClassScores = predClassScores.permute(0, 2, 1).contiguous() # (batchSize, 80 * 80 + 40 * 40 + 20 * 20, nc)

        # ground truth preprocess
        targets = self.preprocess(targets.to(self.mcfg.device), batchSize, scaleTensor=self.model.scaleTensor) # (batchSize, maxCount, 5)
        gtLabels, gtBboxes = targets.split((1, 4), 2)  # cls=(batchSize, maxCount, 1), xyxy=(batchSize, maxCount, 4)
        gtMask = gtBboxes.sum(2, keepdim=True).gt_(0.0)

        # raise NotImplementedError("DetectionLoss::__call__")

        # 1. 生成anchor points（锚点）
        anchorPoints, stridesTensor = self.makeAnchors(preds, self.layerStrides, 0.5)

        # 2. 将预测的分布转换为边界框坐标
        predBboxes = self.distToBbox(predBoxDistribution, anchorPoints.unsqueeze(0))

        # 3. 使用TaskAlignedAssigner进行正负样本分配
        _, targetBboxes, targetScores, fgMask, _ = self.assigner(
            predClassScores.detach().sigmoid(),  # 预测的类别分数（detach防止梯度传播）
            (predBboxes.detach() * stridesTensor).type(gtBboxes.dtype),  # 预测的边界框
            anchorPoints * stridesTensor,  # 锚点坐标
            gtLabels,  # 真实类别标签
            gtBboxes,  # 真实边界框
            gtMask  # 有效目标掩码
        )

        # 4. 计算目标分数总和（用于归一化）
        targetScoresSum = max(targetScores.sum(), 1)

        # 5. 计算分类损失（BCE Loss）
        # 对所有预测进行分类损失计算
        loss[1] = self.bce(predClassScores, targetScores.to(predClassScores.dtype)).sum() / targetScoresSum

        # 6. 计算边界框损失（只对正样本）
        if fgMask.sum():
            # 将预测边界框转换到原始图像尺度
            targetBboxes /= stridesTensor

            # 计算IoU损失和DFL损失
            loss[0], loss[2] = self.bboxLoss(
                predBoxDistribution,  # 预测的分布
                predBboxes,  # 预测的边界框
                anchorPoints,  # 锚点
                targetBboxes,  # 目标边界框
                targetScores,  # 目标分数
                targetScoresSum,  # 目标分数总和
                fgMask  # 前景掩码
            )

        # ===================== 补全部分结束 =====================

        loss[0] *= self.mcfg.lossWeights[0]  # box
        loss[1] *= self.mcfg.lossWeights[1]  # cls
        loss[2] *= self.mcfg.lossWeights[2]  # dfl

        return loss.sum()

    # 需要添加的辅助方法

    def makeAnchors(self, feats, strides, grid_cell_offset=0.5):
        """生成锚点坐标"""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device

        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

        return torch.cat(anchor_points), torch.cat(stride_tensor)


    def distToBbox(self, distance, anchor_points, xywh=True, dim=-1):
        """将分布预测转换为边界框坐标"""
        # distance shape: (batch_size, num_anchors, reg_max * 4)
        # anchor_points shape: (batch_size, num_anchors, 2) 或 (num_anchors, 2)

        # 1. 将分布预测转换为距离值
        # 重塑distance为 (batch_size, num_anchors, 4, reg_max)
        batch_size, num_anchors, total_dim = distance.shape
        reg_max = total_dim // 4
        distance = distance.view(batch_size, num_anchors, 4, reg_max)

        # 2. 使用softmax将分布转换为概率，然后计算期望值
        distance = F.softmax(distance, dim=-1)

        # 3. 计算期望距离值
        proj = torch.arange(reg_max, dtype=distance.dtype, device=distance.device).view(1, 1, 1, -1)
        distance = (distance * proj).sum(dim=-1)  # (batch_size, num_anchors, 4)

        # 4. 分离左上角和右下角的距离
        lt, rb = distance.split(2, dim=-1)  # 各自为 (batch_size, num_anchors, 2)

        # 5. 确保anchor_points的维度匹配
        if anchor_points.dim() == 2:  # (num_anchors, 2)
            anchor_points = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)

        # 6. 计算边界框坐标
        x1y1 = anchor_points - lt  # 左上角
        x2y2 = anchor_points + rb  # 右下角

        if xywh:
            # 转换为中心点+宽高格式
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim=-1)  # xywh
        else:
            return torch.cat((x1y1, x2y2), dim=-1)  # xyxy
