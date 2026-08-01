"""
测试脚本：加载训练好的 LoRA 模型，在 val 集上跑推理看效果
"""

import time
import torch
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
from peft import PeftModel

from dataset import VisualGroundingDataset


def main():
    MODEL_ID = "IDEA-Research/grounding-dino-base"
    LORA_PATH = "./output_grounding_dino_lora"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_SAMPLES = 5

    print(f"[Device] {DEVICE}")
    print(f"[Config] samples={NUM_SAMPLES}")

    # --- 加载处理器 ---
    print("[Load] processor ...")
    processor = GroundingDinoProcessor.from_pretrained(MODEL_ID)

    # --- 加载基础模型 + LoRA ---
    print("[Load] base model ...")
    base_model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID)
    base_model.config.auxiliary_loss = False

    # 修复 get_input_embeddings (PEFT 需要)
    import types
    import torch.nn as nn
    _dummy = nn.Embedding(1, 1)
    base_model.model.get_input_embeddings = types.MethodType(lambda self: _dummy, base_model.model)

    print("[Load] LoRA weights ...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()  # 合并 LoRA 权重，推理更快
    model.to(DEVICE)
    model.eval()
    print("[Model] ready.")

    # --- 加载数据集 (val) ---
    print("[Dataset] loading ...")
    ds = VisualGroundingDataset(processor=processor, split="train", modality="depth", max_samples=100)
    # 用最后几张当测试
    test_indices = list(range(max(0, len(ds) - NUM_SAMPLES), len(ds)))
    print(f"[Dataset] total={len(ds)}, test_indices={test_indices}")

    total_time = 0.0

    for idx in test_indices:
        sample = ds[idx]
        image = sample["image"]
        queries = sample["texts"]
        gt_boxes = sample["boxes"]  # [cx, cy, w, h] 归一化

        print(f"\n{'='*60}")
        print(f"[Test] idx={idx}, image_size={image.size}, queries({len(queries)}):")
        for q in queries:
            print(f"  - {q}")

        # --- 推理 ---
        text_prompt = ". ".join(queries) + "."
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        total_time += elapsed

        # 检查原始 logits（调试用）
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        max_logit = logits.sigmoid().max().item()
        print(f"\n[Debug] max sigmoid={max_logit:.4f}, logits shape={logits.shape}")

        # --- 后处理 ---
        target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)  # (h, w)
        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.2,
            text_threshold=0.15,
            target_sizes=target_sizes,
        )
        result = results[0]
        pred_boxes = result["boxes"]      # [x1, y1, x2, y2] 绝对坐标
        pred_scores = result["scores"]
        pred_labels = result["text_labels"]

        print(f"\n[Results] time={elapsed*1000:.1f}ms, {len(pred_boxes)} detections:")
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            box = [round(x, 1) for x in box.tolist()]
            print(f"  [{score:.3f}] {label} → {box}")

        # --- 与 Ground Truth 对比 ---
        print(f"\n[Ground Truth] {len(gt_boxes)} boxes:")
        for i, (box, query) in enumerate(zip(gt_boxes, queries)):
            cx, cy, w, h = box.tolist()
            iw, ih = image.size
            x1 = (cx - w/2) * iw
            y1 = (cy - h/2) * ih
            x2 = (cx + w/2) * iw
            y2 = (cy + h/2) * ih
            print(f"  [{i}] {query} → [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    avg_time = total_time / len(test_indices)
    print(f"\n{'='*60}")
    print(f"[Summary] 平均推理时间: {avg_time*1000:.1f}ms ({avg_time:.4f}s)")
    print(f"[Summary] FPS: {1/avg_time:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
