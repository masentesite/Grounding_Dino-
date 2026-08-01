"""
Grounding DINO + LoRA 训练入口
"""

import torch
from torch.utils.data import random_split
from transformers import (
    GroundingDinoProcessor,
    TrainingArguments,
    Trainer,
)

from dataset import VisualGroundingDataset, GroundingDetCollator
from model import create_lora_model


class GroundingDinoTrainer(Trainer):
    """模型自带 Hungarian matcher loss，Trainer 只做前向传播"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # 确保 labels 里的 tensor 在正确设备上
        device = next(model.parameters()).device
        for lbl in labels:
            for k in lbl:
                if isinstance(lbl[k], torch.Tensor):
                    lbl[k] = lbl[k].to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    MODEL_ID = "IDEA-Research/grounding-dino-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "./output_grounding_dino_lora"

    # 超参
    BATCH_SIZE = 1  # RTX 5060 8GB，1920×1080 图片只能 batch=1
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 1e-4
    MAX_STEPS = 500
    MAX_SAMPLES = 200
    LORA_R = 8
    LORA_ALPHA = 16
    MODALITY = "depth"

    print(f"[Device] {DEVICE}")
    print(f"[Config] batch={BATCH_SIZE}, accum={GRADIENT_ACCUMULATION}, "
          f"lr={LEARNING_RATE}, steps={MAX_STEPS}, samples={MAX_SAMPLES}")

    # --- 处理器 ---
    processor = GroundingDinoProcessor.from_pretrained(MODEL_ID, local_files_only=True)

    # --- 数据集 ---
    print("[Dataset] Building streaming dataset ...")
    full_dataset = VisualGroundingDataset(
        processor=processor,
        split="train",
        modality=MODALITY,
        max_samples=MAX_SAMPLES,
    )
    val_size = max(1, len(full_dataset) // 10)
    train_ds, val_ds = random_split(
        full_dataset, [len(full_dataset) - val_size, val_size]
    )
    print(f"[Dataset] train={len(train_ds)}, val={len(val_ds)}")

    # --- 模型 ---
    model = create_lora_model(
        model_id=MODEL_ID,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        device=DEVICE,
    )

    # --- 训练参数 ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_steps=MAX_STEPS,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,  # RTX 5060 支持 BF16，省一半显存
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    # --- 训练 ---
    trainer = GroundingDinoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=GroundingDetCollator(processor),
    )

    print("\n[Train] Starting ...")
    trainer.train()

    print(f"\n[Save] LoRA weights → {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("[Done]")


if __name__ == "__main__":
    main()
