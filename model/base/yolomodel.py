import os
import torch
import torch.nn as nn
from misc.log import log
from misc.bbox import makeAnchors
from model.base.backbone import Backbone
from model.base.neck import Neck
from model.base.head import DetectHead
from model.base.swin_transformer import SwinTransformer
from model.base.ema import ModelEMA


class YoloModelPhaseSetup(object):
    @staticmethod
    def getModelWRN(phase):
        """
        Reference: resources/yolov8.jpg
        @return (w, r, n), note that n = 3 x d
        """
        if phase == "nano":
            return (0.25, 2, 1)
        if phase == "small":
            return (0.5, 2, 1)
        if phase == "medium":
            return (0.75, 1.5, 2)
        if phase == "large":
            return (1, 1, 3)
        if phase == "extended":
            return (1.25, 1, 3)
        raise ValueError("Invalid model phase: {}".format(phase))


class YoloModel(nn.Module):
    @classmethod
    def loadModelFromFile(cls, mcfg, modelFile):
        if modelFile is None:
            raise ValueError("Model file not set")
        if not os.path.exists(modelFile):
            raise ValueError("Model file not exists")
        model = cls(mcfg)
        model.load(modelFile)
        return model

    def __init__(self, mcfg):
        super(YoloModel, self).__init__()

        self.mcfg = mcfg
        self.inferenceMode = False

        # model layes
        w, r, n = YoloModelPhaseSetup.getModelWRN(mcfg.phase)
        if self.mcfg.useBone == "backbone":
            self.backbone = Backbone(w, r, n)
        elif self.mcfg.useBone == "swin":
            self.swinTransformer = SwinTransformer(w, r, n)
            log.inf("Using Swin Transformer as backbone...")
        self.neck = Neck(w, r, n)
        self.head = DetectHead(w, r, self.mcfg.nc, self.mcfg.regMax)

        # EMA initialization
        if hasattr(mcfg, 'useEMA') and mcfg.useEMA:
            self.ema = ModelEMA(
                model=self,
                decay=getattr(mcfg, 'ema_decay', 0.9999),
                updates=0
            )
            if hasattr(self.mcfg, 'device') and self.ema:
                 self.ema.get_model_for_eval().to(self.mcfg.device)

        # model static data
        self.layerStrides = [8, 16, 32]
        self.outputShapes = (
            (self.mcfg.nc + self.mcfg.regMax * 4, int(self.mcfg.inputShape[0] / 8), int(self.mcfg.inputShape[1] / 8)),
            (self.mcfg.nc + self.mcfg.regMax * 4, int(self.mcfg.inputShape[0] / 16), int(self.mcfg.inputShape[1] / 16)),
            (self.mcfg.nc + self.mcfg.regMax * 4, int(self.mcfg.inputShape[0] / 32), int(self.mcfg.inputShape[1] / 32)),
        )
        self.anchorPoints, self.anchorStrides = makeAnchors([x[-2:] for x in self.outputShapes], self.layerStrides, 0.5)
        self.anchorPoints = self.anchorPoints.to(self.mcfg.device)
        self.anchorStrides = self.anchorStrides.to(self.mcfg.device)
        self.proj = torch.arange(self.mcfg.regMax, dtype=torch.float).to(self.mcfg.device)
        self.scaleTensor = torch.tensor(self.mcfg.inputShape, device=self.mcfg.device, dtype=torch.float)[[1, 0, 1, 0]]

    def getTrainLoss(self):
        from train.loss import DetectionLoss
        return DetectionLoss(self.mcfg, self)

    def setInferenceMode(self, flag):
        self.inferenceMode = flag
        if self.inferenceMode:
            self.eval()
        else:
            self.train()
        return self

    def freezeBackbone(self):
        log.inf("Freezing backbone parameters...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        return self
        # raise NotImplementedError("YoloModel::freezeBackbone")

    def unfreezeBackbone(self):
        log.inf("Unfreezing backbone parameters...")
        for param in self.backbone.parameters():
            param.requires_grad = True
        return self
        # raise NotImplementedError("YoloModel::unfreezeBackbone")

    def freezeSwinTransformer(self):
        log.inf("Freezing Swin Transformer parameters...")
        for param in self.swinTransformer.parameters():
            param.requires_grad = False
        return self

    def unfreezeSwinTransformer(self):
        log.inf("Unfreezing Swin Transformer parameters...")
        for param in self.swinTransformer.parameters():
            param.requires_grad = True
        return self

    def forward(self, x):
        if self.inferenceMode:
            with torch.no_grad():
                return self.forwardInternal(x)
        return self.forwardInternal(x)

    def forwardInternal(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            xo: (B, nc + regMax * 4, 80, 80)
            yo: (B, nc + regMax * 4, 40, 40)
            zo: (B, nc + regMax * 4, 20, 20)
        """
        if self.mcfg.useBone == "backbone":
            _, feat1, feat2, feat3 = self.backbone.forward(x)
        elif self.mcfg.useBone == "swin":
            _, feat1, feat2, feat3 = self.swinTransformer.forward(x)
        _, X, Y, Z = self.neck.forward(feat1, feat2, feat3)
        xo, yo, zo = self.head.forward(X, Y, Z)
        return xo, yo, zo

    def save(self, modelFile, verbose=False):
        torch.save(self.state_dict(), modelFile)
        if verbose:
            log.inf("Yolo model state saved at {}".format(modelFile))

    def loadBackboneWeights(self, url):
        pretrainedState = torch.hub.load_state_dict_from_url(
            url=url,
            map_location="cpu",
            model_dir=self.mcfg.downloadDir(),
            progress=False,
        )
        missingKeys, unexpectedKeys = self.backbone.load_state_dict(pretrainedState, strict=False)
        if len(unexpectedKeys) > 0:
            log.yellow("Unexpected keys found in model url, ignored:\nunexpected={}\nurl={}".format(unexpectedKeys, url))
        if len(missingKeys) > 0:
            log.red("Missing keys in model url:\nmissing={}\nurl={}".format(missingKeys, url))
            # import pdb; pdb.set_trace()
        else:
            log.grey("Pretrained backbone weights loaded from url: {}".format(url))

    def load(self, modelFile):
        modelState = torch.load(modelFile, weights_only=True)
        missingKeys, unexpectedKeys = self.load_state_dict(modelState, strict=False)
        if len(unexpectedKeys) > 0:
            log.yellow("Unexpected keys found in model file, ignored:\nunexpected={}\nurl={}".format(unexpectedKeys, modelFile))
        if len(missingKeys) > 0:
            log.red("Missing keys in model file:\nmissing={}\nurl={}".format(missingKeys, modelFile))
            # import pdb; pdb.set_trace()
        else:
            log.grey("Yolo model loaded from file: {}".format(modelFile))
