import torch
from overrides import override # this could be removed since Python 3.12
from .loss import DetectionLoss, CWDLoss,  ResponseLoss

class DistillationDetectionLoss(object):
    def __init__(self, mcfg, model):
        self.mcfg = mcfg
        self.histMode = False
        self.detectionLoss = DetectionLoss(mcfg, model)
        # self.cwdLoss = CWDLoss(self.mcfg.device)
        cwd_temperature = getattr(mcfg, 'cwd_temperature', 1.0)  # 从mcfg获取温度，若无则默认为1.0
        cwd_spatial_kl_reduction = getattr(mcfg, 'cwd_spatial_kl_reduction', 'batchmean') # 从mcfg获取，若无则用默认值

        self.cwdLoss = CWDLoss(
            temperature=cwd_temperature,
            spatial_kl_reduction=cwd_spatial_kl_reduction
        )
        self.respLoss = ResponseLoss(self.mcfg.device, self.mcfg.nc, self.mcfg.teacherClassIndexes, self.mcfg.regMax)

        # raise NotImplementedError("DistillationDetectionLoss::__init__")


    @override
    def __call__(self, rawPreds, batch):
        """
        rawPreds[0] & rawPreds[1] shape: (
            (B, regMax * 4 + nc, 80, 80),
            (B, regMax * 4 + nc, 40, 40),
            (B, regMax * 4 + nc, 20, 20),
            (B, 128 * w, 160, 160),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
            (B, 512 * w, 40, 40),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
        )
        """
        spreds = rawPreds[0]
        tpreds = rawPreds[1]

        sresponse, sfeats = spreds[:3], spreds[3:]
        tresponse, tfeats = tpreds[:3], tpreds[3:]

        loss = torch.zeros(3, device=self.mcfg.device)  # original, cwd distillation, response distillation
        loss[0] = self.detectionLoss(sresponse, batch) * self.mcfg.distilLossWeights[0]  # original
        loss[1] = self.cwdLoss(sfeats, tfeats) * self.mcfg.distilLossWeights[1]  # cwd distillation
        loss[2] = self.respLoss(sresponse, tresponse) * self.mcfg.distilLossWeights[2]  # response distillation

        return loss.sum()
