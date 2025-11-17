# ============================================================
# DINOv2 + DETR + Flan-T5 on Pascal VOC 2012 (KaggleHub)
# ============================================================
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

import kagglehub
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm
from timm.data import resolve_model_data_config, create_transform
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    AutoTokenizer,
    T5ForConditionalGeneration,
)

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 3
BATCH_SIZE = 2
LR = 1e-5

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa",
    "train","tvmonitor"
]
CLASS2ID = {c: i for i, c in enumerate(VOC_CLASSES)}
ID2CLASS = {i: c for c, i in CLASS2ID.items()}

# ============================================================
# 1. DOWNLOAD DATASET & AUTO-DETECT VOC PATH
# ============================================================
print("Downloading Pascal VOC 2012 via KaggleHub...")
voc_root = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")
print("Downloaded to:", voc_root)

VOC_PATH = None
for root, dirs, files in os.walk(voc_root):
    if "JPEGImages" in dirs and "Annotations" in dirs:
        VOC_PATH = root
        break

if VOC_PATH is None:
    raise RuntimeError(
        "Could not find Pascal VOC folder (with JPEGImages + Annotations). "
        "Inspect voc_root manually."
    )

print("\nResolved VOC_PATH =", VOC_PATH)

IMAGES_DIR = os.path.join(VOC_PATH, "JPEGImages")
ANNOTS_DIR = os.path.join(VOC_PATH, "Annotations")
IMAGESETS_MAIN = os.path.join(VOC_PATH, "ImageSets", "Main")

# If ImageSets/Main missing → create simple 80/20 train/val split
if not os.path.exists(IMAGESETS_MAIN):
    print("ImageSets/Main not found → creating custom train/val split...")
    os.makedirs(IMAGESETS_MAIN, exist_ok=True)

    all_imgs = sorted(
        [f[:-4] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")]
    )
    if len(all_imgs) == 0:
        raise RuntimeError("No .jpg images found in JPEGImages folder.")

    split = int(0.8 * len(all_imgs))
    train_ids = all_imgs[:split]
    val_ids   = all_imgs[split:]

    with open(os.path.join(IMAGESETS_MAIN, "train.txt"), "w") as f:
        f.write("\n".join(train_ids))
    with open(os.path.join(IMAGESETS_MAIN, "val.txt"), "w") as f:
        f.write("\n".join(val_ids))

TRAIN_SPLIT_FILE = os.path.join(IMAGESETS_MAIN, "train.txt")
VAL_SPLIT_FILE   = os.path.join(IMAGESETS_MAIN, "val.txt")

print("Train split file:", TRAIN_SPLIT_FILE)
print("Val   split file:", VAL_SPLIT_FILE)



# ============================================================
# 2. DATASET CLASS (COCO-STYLE ANNOTATIONS FOR DETR)
# ============================================================
class VOCDataset(Dataset):
    def __init__(self, split_file: str, images_dir: str, annots_dir: str):
        with open(split_file, "r") as f:
            self.ids = [x.strip() for x in f.readlines()]
        self.images_dir = images_dir
        self.annots_dir = annots_dir

    def __len__(self):
        return len(self.ids)

    def _parse_annotation(self, image_id: str) -> Dict[str, Any]:
        xml_path = os.path.join(self.annots_dir, image_id + ".xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotations = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in CLASS2ID:
                continue
            cls_id = CLASS2ID[cls_name]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            w = xmax - xmin
            h = ymax - ymin
            area = w * h

            annotations.append(
                {
                    "bbox": [xmin, ymin, w, h],  # [x, y, w, h]
                    "category_id": cls_id,
                    "area": area,
                    "iscrowd": 0,
                }
            )

        return {"image_id": image_id, "annotations": annotations}

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = os.path.join(self.images_dir, image_id + ".jpg")
        image = Image.open(img_path).convert("RGB")
        target = self._parse_annotation(image_id)
        return image, target



    # ============================================================
    # 3. DATALOADERS
    # ============================================================
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)

    train_dataset = VOCDataset(TRAIN_SPLIT_FILE, IMAGES_DIR, ANNOTS_DIR)
    val_dataset   = VOCDataset(VAL_SPLIT_FILE,   IMAGES_DIR, ANNOTS_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"\nTrain images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    # ============================================================
    # 4. MODELS: DINOv2 BACKBONE + DETR DETECTOR + FLAN-T5 LLM
    # ============================================================
    print("\nLoading DINOv2 backbone...")
    dino_model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,   # feature extractor only
    ).to(DEVICE)
    dino_model.eval()

    # Use timm's official transform for this model (handles size etc.)
    data_cfg = resolve_model_data_config(dino_model)
    dino_transform = create_transform(**data_cfg, is_training=False)
    print("DINOv2 data config:", data_cfg)

    # Small head for auxiliary classification (for monitoring)
    dino_head = nn.Linear(dino_model.num_features, len(VOC_CLASSES)).to(DEVICE)

    print("Loading DETR model...")
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(VOC_CLASSES),
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    detr_model.config.id2label = ID2CLASS
    detr_model.config.label2id = CLASS2ID

    print("Loading LLM (Flan-T5-small)...")
    llm_name = "google/flan-t5-small"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm_model = T5ForConditionalGeneration.from_pretrained(llm_name).to(DEVICE)


    # ============================================================
    # 6. HELPER FUNCTIONS
    # ============================================================
    def prepare_batch(images: List[Image.Image], targets: List[Dict[str, Any]]):
        encoding = image_processor(
            images=images,
            annotations=targets,
            return_tensors="pt",
        )
        pixel_values = encoding["pixel_values"].to(DEVICE)
        pixel_mask = encoding["pixel_mask"].to(DEVICE)
        labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in encoding["labels"]]
        return pixel_values, pixel_mask, labels

    @torch.no_grad()
    def compute_global_features(images: List[Image.Image]) -> torch.Tensor:
        # Use timm transform (ensures correct size / normalization)
        tensors = [dino_transform(img).to(DEVICE) for img in images]
        batch = torch.stack(tensors)  # (B, C, H, W)
        feats = dino_model(batch)     # (B, D)
        return feats