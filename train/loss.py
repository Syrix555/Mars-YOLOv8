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
            # h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

        return torch.cat(anchor_points), torch.cat(stride_tensor)


    def distToBbox(self, distance, anchor_points, xywh=False, dim=-1):
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

class CWDLoss(nn.Module):
    def __init__(self, device): # 只接收 device 参数
        super().__init__()
        self.device = device # 存储 device
        # spatial_kl_reduction 在内部固定，例如使用 'batchmean'
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        # print(f"CWDLoss 初始化参数: device='{device}', "
        #       f"spatial_kl_reduction='batchmean' (hardcoded)")

    def _resize_if_needed(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            teacher_feat = F.interpolate(teacher_feat,
                                         size=student_feat.shape[2:],
                                         mode='bilinear',
                                         align_corners=False)
        return teacher_feat

    def forward(self,
                student_features_list: list[torch.Tensor],
                teacher_features_list: list[torch.Tensor]) -> torch.Tensor:
        current_device = self.device
        if student_features_list and student_features_list[0] is not None:
            current_device = student_features_list[0].device
        elif teacher_features_list and teacher_features_list[0] is not None:
            current_device = teacher_features_list[0].device

        if not student_features_list:
            return torch.tensor(0.0, device=current_device)

        total_cwd_loss = torch.tensor(0.0, device=current_device)

        if not teacher_features_list:
            return total_cwd_loss

        if len(student_features_list) != len(teacher_features_list):
            return total_cwd_loss

        num_valid_levels = 0
        for s_feat, t_feat in zip(student_features_list, teacher_features_list):
            if s_feat is None or t_feat is None:
                continue
            if s_feat.ndim != 4 or t_feat.ndim != 4 or s_feat.numel() == 0 or t_feat.numel() == 0:
                continue

            B_s, C_s, H_s, W_s = s_feat.shape
            B_t, C_t, _, _ = t_feat.shape

            if B_s != B_t: continue
            if C_s != C_t: continue
            if C_s == 0: continue
            if H_s * W_s == 0: continue

            t_feat_aligned = self._resize_if_needed(s_feat, t_feat)
            s_feat_flat = s_feat.view(B_s, C_s, H_s * W_s)
            t_feat_flat = t_feat_aligned.view(B_t, C_s, H_s * W_s)

            log_student_spatial_dist = F.log_softmax(s_feat_flat, dim=2)
            teacher_spatial_dist = F.softmax(t_feat_flat, dim=2)

            kl_loss_per_level = self.kl_div_loss(
                log_student_spatial_dist.view(-1, H_s * W_s),
                teacher_spatial_dist.view(-1, H_s * W_s)
            )
            total_cwd_loss += kl_loss_per_level
            num_valid_levels += 1

        if num_valid_levels > 0:
            final_loss = total_cwd_loss / num_valid_levels
        else:
            final_loss = total_cwd_loss
        return final_loss

class KLDivergenceLoss(nn.Module):
    def __init__(self, temperature=1.0, reduction='batchmean'):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        # print(f"KLDivergenceLoss initialized with temperature={temperature}, reduction='{reduction}'")

    def forward(self, student_logits, teacher_logits):
        if student_logits.numel() == 0 or teacher_logits.numel() == 0:
            return torch.tensor(0.0, device=student_logits.device if student_logits.numel() > 0 else teacher_logits.device)

        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        log_soft_student_outputs = F.log_softmax(student_logits / self.temperature, dim=-1)
        loss = self.kl_div(log_soft_student_outputs, soft_targets)
        return loss * (self.temperature ** 2)

class ResponseLoss(nn.Module):
    """
    计算学生模型和教师模型在分类响应上的KL散度损失。
    响应通常是来自检测头不同层级的特征图。
    """
    def __init__(self, device, nc, teacher_class_indexes, reg_max, temperature=1.0, reduction='batchmean'):
        """
        初始化 ResponseLoss 模块。

        参数:
            device (torch.device): 张量操作的设备。
            nc (int): 类别数量。
            teacher_class_indexes (list 或 None): 要进行蒸馏的教师模型的类别索引列表。
                                                  如果为 None，则使用所有类别。
            reg_max (int): 用于边界框回归的正则化最大值，用于确定通道分割。
            temperature (float, 可选): KL散度的温度。默认为 1.0。
            reduction (str, 可选): KL散度的规约方法 ('batchmean', 'sum'等)。
                                      默认为 'batchmean'。
        """
        super().__init__()
        self.device = device
        self.nc = nc
        self.reg_max_channels = reg_max * 4  # bounding box 分布的通道数

        # 处理 teacherClassIndexes
        self.teacher_class_indexes = teacher_class_indexes
        if self.teacher_class_indexes is not None:
            if not isinstance(self.teacher_class_indexes, torch.Tensor):
                self.teacher_class_indexes = torch.tensor(self.teacher_class_indexes, dtype=torch.long)
            # teacher_class_indexes 会在 forward 中根据 logits 的设备进行 .to(device) 操作

        # 实例化 KLDivergenceLoss
        self.kl_loss_calculator = KLDivergenceLoss(temperature=temperature, reduction=reduction)

    def forward(self, student_responses_list, teacher_responses_list):
        """
        计算基于响应的蒸馏损失。

        参数:
            student_responses_list (list of torch.Tensor): 学生模型的响应张量列表 (来自检测头)。
                每个张量的形状: (Batch, regMax*4 + NumClasses, Height, Width)。
            teacher_responses_list (list of torch.Tensor): 教师模型的响应张量列表，格式相同。

        返回:
            torch.Tensor: 在所有有效层级上计算的平均KL散度损失。
        """
        total_kl_loss = torch.tensor(0.0, device=self.device)
        valid_levels_count = 0

        for s_resp, t_resp in zip(student_responses_list, teacher_responses_list):
            if s_resp is None or t_resp is None or s_resp.numel() == 0 or t_resp.numel() == 0:
                continue

            # s_resp 形状: (B, C, H, W)，其中 C = reg_max_channels + nc
            # 从组合的响应张量中提取类别 logits 部分
            try:
                # 类别 logits 是最后 'self.nc' 个通道
                s_class_logits_map = s_resp.split((self.reg_max_channels, self.nc), dim=1)[1]
                t_class_logits_map = t_resp.split((self.reg_max_channels, self.nc), dim=1)[1]
            except RuntimeError as e:
                print(f"警告: ResponseLoss 无法分割响应通道。学生响应形状: {s_resp.shape}, "
                      f"教师响应形状: {t_resp.shape}, 期望分割: ({self.reg_max_channels}, {self.nc})。错误: {e}")
                continue

            # 为 KLDivergenceLoss 重塑形状: (Batch*Height*Width, NumClasses)
            b, _, h, w = s_class_logits_map.shape
            s_logits_flat = s_class_logits_map.permute(0, 2, 3, 1).contiguous().view(-1, self.nc)
            t_logits_flat = t_class_logits_map.permute(0, 2, 3, 1).contiguous().view(-1, self.nc)

            current_target_device = s_logits_flat.device # 确保索引与 logits 在同一设备上

            # 如果指定了 teacher_class_indexes，则应用它
            if self.teacher_class_indexes is not None:
                if self.teacher_class_indexes.device != current_target_device:
                    self.teacher_class_indexes = self.teacher_class_indexes.to(current_target_device)

                s_logits_selected = s_logits_flat[:, self.teacher_class_indexes]
                t_logits_selected = t_logits_flat[:, self.teacher_class_indexes]
            else:
                s_logits_selected = s_logits_flat
                t_logits_selected = t_logits_flat

            if s_logits_selected.numel() == 0 or t_logits_selected.numel() == 0:
                continue

            level_loss = self.kl_loss_calculator(s_logits_selected, t_logits_selected)
            total_kl_loss += level_loss
            valid_levels_count += 1

        if valid_levels_count > 0:
            return total_kl_loss / valid_levels_count
        else:
            return total_kl_loss # 如果没有有效层级，则返回 0.0
