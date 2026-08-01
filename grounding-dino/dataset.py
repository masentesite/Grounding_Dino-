"""
数据集: masentesite/visual-grounding-3modal
流式加载 COCO 标注 + ModelScope API 按需加载图片
"""

import json
import requests
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import GroundingDinoProcessor


class VisualGroundingDataset(Dataset):
    """
    流式数据集：图片通过 ModelScope API 按需加载，标注从 COCO JSON 解析。
    """

    BASE_URL = "https://modelscope.cn/api/v1/datasets/masentesite/visual-grounding-3modal/repo"

    def __init__(
        self,
        processor: GroundingDinoProcessor,
        split: str = "train",
        modality: str = "depth",
        max_samples: Optional[int] = None,
    ):
        self.processor = processor
        self.modality = modality
        self.split = split

        self.annotations = self._load_annotations()
        self.images_meta, self.annotations_meta = self._filter_by_modality()

        self.image_map: Dict[int, dict] = {}
        for img in self.images_meta:
            self.image_map[img["id"]] = {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "annotations": [],
                "queries": [],
                "boxes": [],
            }
        for ann in self.annotations_meta:
            if ann["image_id"] in self.image_map:
                self.image_map[ann["image_id"]]["annotations"].append(ann)
                self.image_map[ann["image_id"]]["queries"].append(ann["query"])
                # COCO [x, y, w, h] → 归一化 [cx, cy, w, h]
                x, y, w, h = ann["bbox"]
                iw = self.image_map[ann["image_id"]]["width"]
                ih = self.image_map[ann["image_id"]]["height"]
                self.image_map[ann["image_id"]]["boxes"].append([
                    (x + w / 2) / iw,
                    (y + h / 2) / ih,
                    w / iw,
                    h / ih,
                ])

        self.image_ids = [
            iid for iid, info in self.image_map.items() if info["annotations"]
        ]

        # 过滤文本过长导致截断后标签不匹配的样本
        valid_ids = []
        for iid in self.image_ids:
            info = self.image_map[iid]
            text_str = ". ".join(info["queries"]) + "."
            tokens = self.processor.tokenizer(text_str, add_special_tokens=True)
            if len(tokens["input_ids"]) <= 256:
                valid_ids.append(iid)
        n_filtered = len(self.image_ids) - len(valid_ids)
        self.image_ids = valid_ids
        if n_filtered > 0:
            print(f"[Dataset] Filtered {n_filtered} images with text > 256 tokens")
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]

        total_boxes = sum(len(self.image_map[i]["annotations"]) for i in self.image_ids)
        print(f"[Dataset] {len(self.image_ids)} images, {total_boxes} boxes")

    def _load_annotations(self):
        url = f"{self.BASE_URL}?Source=SDK&Revision=master&FilePath=coco_unified.json"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return json.loads(resp.text)

    def _filter_by_modality(self):
        images = [
            img for img in self.annotations["images"]
            if img["modality"] == self.modality
            and img.get("split", "train") == self.split
        ]
        image_ids = {img["id"] for img in images}
        annotations = [
            ann for ann in self.annotations["annotations"]
            if ann["image_id"] in image_ids
        ]
        return images, annotations

    def _load_image(self, file_name: str) -> Image.Image:
        url = f"{self.BASE_URL}?Source=SDK&Revision=master&FilePath=data%2F{file_name.replace('/', '%2F')}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.image_map[img_id]
        image = self._load_image(info["file_name"])
        return {
            "image": image,
            "texts": info["queries"],
            "boxes": torch.tensor(info["boxes"], dtype=torch.float32),
            "image_id": img_id,
        }


@dataclass
class GroundingDetCollator:
    """batch → model inputs + labels"""
    processor: GroundingDinoProcessor

    def __call__(self, batch: List[dict]) -> dict:
        images = [item["image"] for item in batch]
        all_texts = []
        all_boxes = []

        for item in batch:
            text_str = " . ".join(item["texts"]) + " ."
            all_texts.append(text_str)
            all_boxes.append(item["boxes"])

        inputs = self.processor(
            images=images, text=all_texts,
            return_tensors="pt", padding=True,
            max_length=256, truncation=True,
        )

        labels = []
        for boxes in all_boxes:
            n = len(boxes)
            if n > 0:
                labels.append({
                    "boxes": boxes.clone(),
                    "class_labels": torch.arange(n, dtype=torch.long),
                })
            else:
                labels.append({
                    "boxes": torch.zeros((0, 4)),
                    "class_labels": torch.zeros(0, dtype=torch.long),
                })

        return {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"],
            "token_type_ids": inputs.get("token_type_ids"),
            "attention_mask": inputs["attention_mask"],
            "pixel_mask": inputs["pixel_mask"],
            "labels": labels,
        }
