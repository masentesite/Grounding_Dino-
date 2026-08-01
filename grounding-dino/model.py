"""
Grounding DINO Base + LoRA 模型
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import GroundingDinoForObjectDetection


def create_lora_model(
    model_id: str = "IDEA-Research/grounding-dino-base",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    device: str = "cuda",
) -> nn.Module:
    """
    加载 Grounding DINO Base，注入 LoRA 到 attention 投影层，
    检测头 (bbox_embed, class_embed) 保持全量训练。
    """
    print(f"[Model] Loading {model_id} ...")
    model = GroundingDinoForObjectDetection.from_pretrained(
        model_id, local_files_only=True,
    )

    # 微调初期 loss ~12 万是正常的（DETR encoder 预测 900 个框 vs ~7 个真值），
    # 训几步就会降到几百。关掉 auxiliary_loss 减少一点噪音。
    model.config.auxiliary_loss = False

    # Grounding DINO 是多模态模型，PreTrainedModel.get_input_embeddings
    # 会抛 NotImplementedError，PEFT 初始化 _check_tied_modules 需要它。
    # 直接覆盖 model.model (GroundingDinoModel) 的方法，因为它才是 base_model
    _dummy = nn.Embedding(1, 1)
    import types
    model.model.get_input_embeddings = types.MethodType(lambda self: _dummy, model.model)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "query", "key", "value",
            "out_proj", "output_proj", "dense",
            "fc1", "fc2",
            "vision_proj", "text_proj",
            "values_vision_proj", "values_text_proj",
            "out_vision_proj", "out_text_proj",
            "sampling_offsets", "attention_weights", "value_proj",
        ],
        modules_to_save=[
            *[f"bbox_embed.{i}" for i in range(6)],
            *[f"class_embed.{i}" for i in range(6)],
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    return model
