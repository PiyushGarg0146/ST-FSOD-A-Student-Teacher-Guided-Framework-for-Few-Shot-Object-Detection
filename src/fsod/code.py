
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
# 5. OPTIMIZER
# ============================================================
params = list(detr_model.parameters()) + list(dino_head.parameters())
optimizer = torch.optim.AdamW(params, lr=LR)

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

# ============================================================
# 7. TRAIN & VALIDATION FUNCTIONS
# ============================================================
def train_one_epoch(epoch: int):
    detr_model.train()
    dino_head.train()

    epoch_loss = 0.0
    epoch_aux_correct = 0
    epoch_aux_total = 0

    for step, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        pixel_values, pixel_mask, labels = prepare_batch(images, targets)
        outputs = detr_model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )
        loss = outputs.loss  # DETR loss

        # DINOv2 auxiliary classification
        feats = compute_global_features(images)
        logits = dino_head(feats)

        aux_targets = []
        for tgt in labels:
            if len(tgt["class_labels"]) > 0:
                vals, counts = tgt["class_labels"].unique(return_counts=True)
                aux_targets.append(int(vals[counts.argmax()].item()))
            else:
                aux_targets.append(0)
        aux_targets = torch.tensor(aux_targets, dtype=torch.long, device=DEVICE)

        aux_loss = nn.CrossEntropyLoss()(logits, aux_targets)
        total_loss = loss + 0.1 * aux_loss

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item() * len(images)

        preds = logits.argmax(dim=1)
        epoch_aux_correct += (preds == aux_targets).sum().item()
        epoch_aux_total += len(aux_targets)

        if (step + 1) % 50 == 0:
            print(
                f"[Epoch {epoch} | Step {step+1}/{len(train_loader)}] "
                f"Loss: {total_loss.item():.4f}"
            )

    avg_loss = epoch_loss / len(train_dataset)
    aux_acc = epoch_aux_correct / max(1, epoch_aux_total)
    return avg_loss, aux_acc

@torch.no_grad()
def validate_epoch(epoch: int):
    detr_model.eval()
    dino_head.eval()

    epoch_loss = 0.0
    all_true = []
    all_pred = []

    for images, targets in val_loader:
        pixel_values, pixel_mask, labels = prepare_batch(images, targets)
        outputs = detr_model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )

        loss = outputs.loss
        epoch_loss += loss.item() * len(images)

        # Post-process predictions
        target_sizes = []
        for img in images:
            w, h = img.size
            target_sizes.append((h, w))
        target_sizes = torch.tensor(target_sizes, device=DEVICE)

        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.5,
        )

        # Simple label-level matching
        for res, tgt in zip(results, labels):
            true_cls = tgt["class_labels"].cpu().numpy().tolist()
            if len(true_cls) == 0:
                continue
            pred_labels = res["labels"].cpu().numpy().tolist()
            n = min(len(true_cls), len(pred_labels))
            all_true.extend(true_cls[:n])
            all_pred.extend(pred_labels[:n])

    avg_loss = epoch_loss / len(val_dataset)

    if len(all_true) > 0:
        report = classification_report(
            all_true,
            all_pred,
            labels=list(range(len(VOC_CLASSES))),
            target_names=VOC_CLASSES,
            output_dict=True,
            zero_division=0,
        )
        overall_acc = report["accuracy"]
    else:
        report = None
        overall_acc = 0.0

    return avg_loss, overall_acc, report, all_true, all_pred

# ============================================================
# 8. TRAINING LOOP
# ============================================================
train_losses = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0
best_state = None
last_val_report = None
last_all_true = []
last_all_pred = []

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")
    train_loss, aux_acc = train_one_epoch(epoch)
    val_loss, val_acc, val_report, all_true, all_pred = validate_epoch(epoch)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    last_val_report = val_report
    last_all_true = all_true
    last_all_pred = all_pred

    print("Epoch", epoch, "summary:")
    print(f"  Train loss        : {train_loss:.4f}")
    print(f"  Val loss          : {val_loss:.4f}")
    print(f"  Aux DINO accuracy : {aux_acc:.4f}")
    print(f"  Val detection acc : {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {
            "epoch": epoch,
            "detr_model": detr_model.state_dict(),
            "dino_head": dino_head.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

if best_state is not None:
    torch.save(best_state, "best_dinov2_detr_llm_voc2012.pth")
    print(
        f"\nBest model saved from epoch {best_state['epoch']} "
        f"with val acc {best_val_acc:.4f}"
    )

# ============================================================
# 9. RESULT, EVALUATION, SUMMARY MATRICES
# ============================================================
results_df = pd.DataFrame(
    {
        "epoch": list(range(1, NUM_EPOCHS + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies,
    }
)
print("\n=== RESULT MATRIX (Per-epoch Metrics) ===")
print(results_df)
results_df.to_csv("result_matrix_epochs.csv", index=False)

if last_val_report is not None:
    class_rows = []
    for cls_name in VOC_CLASSES:
        if cls_name in last_val_report:
            r = last_val_report[cls_name]
            class_rows.append(
                {
                    "class": cls_name,
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1-score": r["f1-score"],
                    "support": r["support"],
                }
            )
    eval_df = pd.DataFrame(class_rows)
else:
    eval_df = pd.DataFrame()

print("\n=== EVALUATION MATRIX (Per-class Metrics) ===")
print(eval_df)
eval_df.to_csv("evaluation_matrix_per_class.csv", index=False)

if last_val_report is not None:
    summary_row = {
        "accuracy": last_val_report.get("accuracy", 0.0),
        "macro_precision": last_val_report["macro avg"]["precision"],
        "macro_recall": last_val_report["macro avg"]["recall"],
        "macro_f1": last_val_report["macro avg"]["f1-score"],
        "weighted_precision": last_val_report["weighted avg"]["precision"],
        "weighted_recall": last_val_report["weighted avg"]["recall"],
        "weighted_f1": last_val_report["weighted avg"]["f1-score"],
    }
    summary_df = pd.DataFrame([summary_row])
else:
    summary_df = pd.DataFrame()

print("\n=== SUMMARY MATRIX (Overall Metrics) ===")
print(summary_df)
summary_df.to_csv("summary_matrix_overall.csv", index=False)

if len(last_all_true) > 0:
    cm = confusion_matrix(
        last_all_true, last_all_pred, labels=list(range(len(VOC_CLASSES)))
    )
    cm_df = pd.DataFrame(cm, index=VOC_CLASSES, columns=VOC_CLASSES)
    print("\n=== CONFUSION MATRIX ===")
    print(cm_df)
    cm_df.to_csv("confusion_matrix.csv")
else:
    cm = None
    print("\nNo predictions to build confusion matrix.")

# ============================================================
# 10. GRAPHS (LOSS, VAL LOSS, ACCURACY & CONFUSION HEATMAP)
# ============================================================
epochs = np.arange(1, NUM_EPOCHS + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, marker="o", label="Train Loss")
plt.plot(epochs, val_losses, marker="s", label="Val Loss")
plt.plot(epochs, val_accuracies, marker="^", label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training & Validation Metrics per Epoch")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("loss_val_acc_curves.png", dpi=200)
plt.close()
print("\nSaved combined metrics graph: loss_val_acc_curves.png")

if cm is not None:
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix Heatmap")
    plt.colorbar()
    tick_marks = np.arange(len(VOC_CLASSES))
    plt.xticks(tick_marks, VOC_CLASSES, rotation=90)
    plt.yticks(tick_marks, VOC_CLASSES)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("confusion_matrix_heatmap.png", dpi=200)
    plt.close()
    print("Saved confusion matrix heatmap: confusion_matrix_heatmap.png")

# ============================================================
# 11. LLM-GENERATED SUMMARY
# ============================================================
if not summary_df.empty:
    acc_val = float(summary_df["accuracy"].iloc[0])
    macro_f1 = float(summary_df["macro_f1"].iloc[0])
else:
    acc_val = 0.0
    macro_f1 = 0.0

summary_prompt = f"""
You are summarizing an object detection experiment.

Overall accuracy: {acc_val:.4f}
Macro F1-score: {macro_f1:.4f}

Per-epoch results:
{results_df.to_string(index=False)}

Explain briefly:
- whether the model is learning across epochs,
- how training and validation losses behave,
- how validation accuracy changes,
- any sign of overfitting or underfitting.
"""

inputs = llm_tokenizer(
    summary_prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512,
).to(DEVICE)
outputs = llm_model.generate(**inputs, max_new_tokens=200)
text_summary = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== LLM SUMMARY OF TRAINING ===")
print(text_summary)

print("\n✅ Done: model trained, matrices saved, graphs generated, and LLM summary printed.")

