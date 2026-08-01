"""
端到端测试：加载 LoRA 模型 + 数据集 → 前向 + 反向，验证训练流水线
"""

import torch
from transformers import GroundingDinoProcessor

from dataset import VisualGroundingDataset, GroundingDetCollator
from model import create_lora_model


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "IDEA-Research/grounding-dino-base"
    MAX_SAMPLES = 8  # 少量样本测试

    # 1. 加载处理器和模型
    print("=" * 60)
    print("[1/4] Loading processor and LoRA model ...")
    processor = GroundingDinoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = create_lora_model(MODEL_ID, lora_r=8, lora_alpha=16, device=DEVICE)

    # 2. 加载流式数据集
    print("\n[2/4] Loading streaming dataset ...")
    dataset = VisualGroundingDataset(
        processor=processor, split="train", modality="depth", max_samples=MAX_SAMPLES
    )
    print(f"  Total samples: {len(dataset)}")

    # 3. 构造 batch（缩小图片 + batch=1，适配 8GB 显存）
    print("\n[3/4] Building batch (batch=1) ...")
    collator = GroundingDetCollator(processor)
    sample = dataset[0]
    # 缩小图片节省显存
    sample["image"] = sample["image"].resize((512, 384))
    batch = collator([sample])

    print(f"  pixel_values: {batch['pixel_values'].shape}")
    print(f"  input_ids:    {batch['input_ids'].shape}")
    print(f"  labels:       {len(batch['labels'])} items")
    for i, lbl in enumerate(batch['labels']):
        print(f"    [{i}] boxes={lbl['boxes'].shape}, class_labels={lbl['class_labels'].shape}")

    # 4. 前向 + 反向（fp16 混合精度）
    print("\n[4/4] Forward + backward ...")
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    # labels 里的 tensor 也要移到 GPU
    for lbl in batch["labels"]:
        for k in lbl:
            if isinstance(lbl[k], torch.Tensor):
                lbl[k] = lbl[k].to(DEVICE)
    labels = batch.pop("labels")

    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(**batch, labels=labels)
    loss = outputs.loss
    print(f"  loss: {loss.item():.4f}")

    print("  backward ...")
    loss.backward()

    # 检查梯度
    grad_params = 0
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.norm() > 0:
            grad_params += 1
    print(f"  params with grad: {grad_params}")

    # 检查原参数是否冻结
    frozen_ok = True
    for name, p in model.named_parameters():
        # LoRA 参数（含 lora_）和 modules_to_save 参数应该有 grad
        if "lora_" not in name and "bbox_embed" not in name and "class_embed" not in name:
            if p.grad is not None and p.grad.norm() > 0:
                print(f"  [WARN] unexpected grad on: {name}")
                frozen_ok = False

    print("\n" + "=" * 60)
    if frozen_ok and grad_params > 0:
        print("  [OK] End-to-end test passed!")
        print("  - Model loads with LoRA")
        print("  - Streaming dataset works")
        print("  - Forward pass computes loss")
        print("  - Backward only updates LoRA + detection heads")
        print("  - Original parameters are frozen")
    else:
        print("  [FAIL] Something is wrong, check output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
