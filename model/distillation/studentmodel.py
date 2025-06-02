import torch
from misc.log import log
from overrides import override # this could be removed since Python 3.12
from model.base.yolomodel import YoloModel
from model.distillation.teachermodel import YoloTeacherModel


class YoloStudentModel(YoloModel):
    def initTeacherModel(self):
        return YoloTeacherModel.loadModelFromFile(self.mcfg, self.mcfg.teacherModelFile)

    def __init__(self, mcfg):
        super().__init__(mcfg)
        self.teacherModel = self.initTeacherModel()

    def getTrainLoss(self):
        from train.distilloss import DistillationDetectionLoss
        return DistillationDetectionLoss(self.mcfg, self)

    @override
    def forward(self, x):
        if self.inferenceMode:
            with torch.no_grad():
                if self.mcfg.useBone == "backbone":
                    feat0, feat1, feat2, feat3 = self.backbone.forward(x)
                elif self.mcfg.useBone == "swin":
                    feat0, feat1, feat2, feat3 = self.swinTransformer.forward(x)
                C, X, Y, Z = self.neck.forward(feat1, feat2, feat3)
                xo, yo, zo = self.head.forward(X, Y, Z)
                return xo, yo, zo
            
        if self.mcfg.useBone == "backbone":
            feat0, feat1, feat2, feat3 = self.backbone.forward(x)
        elif self.mcfg.useBone == "swin":
            feat0, feat1, feat2, feat3 = self.swinTransformer.forward(x)
        C, X, Y, Z = self.neck.forward(feat1, feat2, feat3)
        xo, yo, zo = self.head.forward(X, Y, Z)
        tlayerOutput = self.teacherModel.forward(x)
        return (xo, yo, zo, feat0, feat1, feat2, feat3, C, X, Y, Z), tlayerOutput

    @override
    def load(self, modelFile):
        """
        Load states except "self.teacherModel"
        """
        selfState = self.state_dict()
        loadedState = torch.load(modelFile, weights_only=True)
        selfState.update(loadedState)
        missingKeys, unexpectedKeys = self.load_state_dict(selfState, strict=False)
        if len(unexpectedKeys) > 0:
            log.yellow("Unexpected keys found in model file, ignored:\nunexpected={}\nurl={}".format(unexpectedKeys, modelFile))
        if len(missingKeys) > 0:
            log.red("Missing keys in model file:\nmissing={}\nurl={}".format(missingKeys, modelFile))
            # import pdb; pdb.set_trace()
        else:
            log.grey("Yolo student model loaded from file: {}".format(modelFile))
