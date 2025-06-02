import copy
import torch
import torch.nn as nn
from misc.log import log

class ModelEMA:
    """
    模型指数移动平均 (Model Exponential Moving Average)
    对模型的状态字典（参数和缓冲区）进行指数移动平均。
    """
    def __init__(self, model, decay=0.9999, updates=0):
        """
        初始化EMA。

        参数:
            model (nn.Module): 需要进行EMA操作的模型。
            decay (float): EMA衰减率。较高的decay值意味着历史权重影响更大，更新更平滑。
            updates (int): 初始的EMA更新次数。
        """
        # 创建EMA模型，它是原始模型的一个深拷贝，并且设置为评估模式
        self.ema_model = copy.deepcopy(model).eval()
        self.updates = updates  # EMA更新的次数

        # 定义衰减率函数，可以根据更新次数动态调整衰减率（可选策略）
        # 这里的2000是一个时间常数，可以根据需要调整
        self.decay_fn = lambda x: decay * (1 - torch.exp(-torch.tensor(x / 2000.0, dtype=torch.float32)))
        # 或者使用固定的衰减率:
        # self.decay_val = decay

        # EMA模型的参数不需要计算梯度
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        current_decay_val = self.decay_fn(self.updates) # 获取初始的实际衰减率
        log.inf(f"EMA: Initialized EMA model with decay {decay:.4f}")

    def update(self, model):
        """
        在每个训练步骤后，用当前训练模型的权重更新EMA模型的权重。

        参数:
            model (nn.Module): 当前训练中的模型。
        """
        with torch.no_grad(): # 更新EMA权重时不需要计算梯度
            self.updates += 1
            current_decay = self.decay_fn(self.updates)
            # current_decay = self.decay_val # 如果使用固定衰减率

            # 获取主训练模型的状态字典
            main_model_state_dict = model.state_dict()

            # 更新EMA模型的每个浮点参数
            for name, ema_param in self.ema_model.state_dict().items():
                if ema_param.dtype.is_floating_point:
                    main_model_param = main_model_state_dict[name].detach() # 获取对应的训练模型参数
                    # EMA更新公式: ema_new = decay * ema_old + (1 - decay) * model_current
                    ema_param.copy_(current_decay * ema_param + (1.0 - current_decay) * main_model_param)

    def get_model_for_eval(self):
        """
        返回EMA模型实例，通常用于评估或最终推理。
        """
        return self.ema_model

    # 你也可以保留 replace_weights 和 restore_weights 方法，如果希望在原始模型上直接操作
    def replace_model_weights(self, target_model):
        """
        用EMA模型的权重替换目标模型的权重。
        通常在评估前调用。

        参数:
            target_model (nn.Module): 需要被EMA权重替换权重的模型。
        """
        log.info("EMA: Replacing model weights with EMA weights...")
        self.original_target_model_state_dict = copy.deepcopy(target_model.state_dict()) # 备份目标模型的原始权重
        target_model.load_state_dict(self.ema_model.state_dict(), strict=True)

    def restore_original_weights(self, target_model):
        """
        恢复目标模型的原始权重（如果之前被replace_model_weights备份过）。
        通常在评估后调用。

        参数:
            target_model (nn.Module): 需要恢复原始权重的模型。
        """
        if hasattr(self, 'original_target_model_state_dict'):
            log.info("EMA: 正在恢复目标模型的原始权重。")
            target_model.load_state_dict(self.original_target_model_state_dict, strict=True)
            del self.original_target_model_state_dict # 删除备份，避免内存占用
        else:
            log.warning("EMA: 未找到用于恢复的原始模型状态字典备份。请确保已先调用 replace_model_weights。")
