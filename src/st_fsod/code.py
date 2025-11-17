
# ============================================================
# 1. IMPORTS & SEEDING
# ============================================================
import os
from pathlib import Path
import random
import numpy as np
from typing import List, Dict, Any

import kagglehub

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import functional as F
from torchvision.ops import box_iou

from lxml import etree
import matplotlib.pyplot as plt
import PIL.Image as Image

from transformers import AutoImageProcessor, AutoModel
from scipy.optimize import linear_sum_assignment

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 2. DATASET LOADING (KaggleHub + optional local folder)
# ============================================================

USE_KAGGLE = True   # set False if you only want to use local uploaded folder
VOC_ROOT = None

if USE_KAGGLE:
    kaggle_path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")
    kaggle_path = Path(kaggle_path)
    print("Kaggle download path:", kaggle_path)

    if (kaggle_path / "Annotations").exists():
        VOC_ROOT = kaggle_path
    else:
        ann_dirs = list(kaggle_path.rglob("Annotations"))
        if len(ann_dirs) == 0:
            raise RuntimeError(
                "Could not find 'Annotations' folder in Kaggle dataset. "
                "Check folder structure or set VOC_ROOT manually."
            )
        VOC_ROOT = ann_dirs[0].parent

# If you uploaded your own VOC2012 folder, override this:
# VOC_ROOT = Path("/content/VOC2012")

if VOC_ROOT is None:
    raise RuntimeError("VOC_ROOT is not set. Please specify your VOC root folder.")

print("Using VOC root:", VOC_ROOT)

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_IDX = {c: i+1 for i, c in enumerate(VOC_CLASSES)}  # 0 = background (unused here)


class VOCDataset(Dataset):
    def __init__(self, root: Path, image_set: str = "train", transforms=None, img_size=800):
        self.root = root
        self.transforms = transforms
        self.img_size = img_size

        split_file = self.root / "ImageSets" / "Main" / f"{image_set}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r") as f:
            self.ids = [x.strip() for x in f.readlines()]

        self.img_dir = self.root / "JPEGImages"
        self.ann_dir = self.root / "Annotations"

    def __len__(self):
        return len(self.ids)

    def parse_voc_xml(self, xml_file: Path):
        tree = etree.parse(str(xml_file))
        root = tree.getroot()

        size = root.find("size")
        w_orig = float(size.find("width").text)
        h_orig = float(size.find("height").text)

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASS_TO_IDX:
                continue
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
            w_orig,
            h_orig,
        )

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.img_dir / f"{img_id}.jpg"
        ann_path = self.ann_dir / f"{img_id}.xml"

        img = Image.open(img_path).convert("RGB")
        boxes, labels, w_orig, h_orig = self.parse_voc_xml(ann_path)

        img_resized = F.resize(img, [self.img_size, self.img_size])
        img_tensor = F.to_tensor(img_resized)  # [3, H, W] in [0,1]

        if boxes.numel() > 0:
            sx = self.img_size / w_orig
            sy = self.img_size / h_orig
            boxes_scaled = boxes.clone()
            boxes_scaled[:, [0, 2]] *= sx
            boxes_scaled[:, [1, 3]] *= sy
        else:
            boxes_scaled = boxes

        target = {
            "boxes": boxes_scaled,   # xyxy in resized image coords
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, list(targets)


img_size = 800
train_dataset = VOCDataset(VOC_ROOT, "train", img_size=img_size)
val_dataset   = VOCDataset(VOC_ROOT, "val",   img_size=img_size)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

# ============================================================
# 3. IMPORTS FOR MODELS (add after your dataset code)
# ============================================================
import timm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
import torch.nn.functional as Fnn  # keep F for transforms.functional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 4. TEACHER MODEL: DINOv2 + DETR-STYLE HEAD
# ============================================================

dino_name = "facebook/dinov2-base"
image_processor = AutoImageProcessor.from_pretrained(dino_name)
dino_backbone = AutoModel.from_pretrained(dino_name)
dino_backbone.eval()
for p in dino_backbone.parameters():
    p.requires_grad = False  # teacher backbone frozen

teacher_backbone_dim = dino_backbone.config.hidden_size  # e.g., 768


class SimpleDETRHead(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=8,
        num_queries=100,
        num_classes=21,      # 20 VOC + 1 background
        num_decoder_layers=3
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed  = nn.Linear(d_model, 4)  # normalized cx, cy, w, h

    def forward(self, memory):
        B = memory.size(0)
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, D]
        hs = self.decoder(tgt=queries, memory=memory)  # [B, Q, D]

        pred_logits = self.class_embed(hs)
        pred_boxes  = self.bbox_embed(hs).sigmoid()
        return pred_logits, pred_boxes


class DINOv2DETR(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images):
        """
        images: [B, 3, H, W] in [0,1]
        """
        # images already in [0,1], so don't rescale again
        inputs = image_processor(
            images=list(images),
            return_tensors="pt",
            do_rescale=False
        )
        pixel_values = inputs["pixel_values"].to(images.device)

        with torch.no_grad():  # teacher backbone is frozen
            outputs = self.backbone(pixel_values=pixel_values)
        memory = outputs.last_hidden_state  # [B, S, D]

        pred_logits, pred_boxes = self.head(memory)
        return pred_logits, pred_boxes


teacher_head = SimpleDETRHead(
    d_model=teacher_backbone_dim,
    nhead=8,
    num_queries=100,
    num_classes=len(VOC_CLASSES) + 1,
    num_decoder_layers=3
)

teacher_model = DINOv2DETR(dino_backbone, teacher_head).to(device)

# Freeze entire teacher (you can also choose to fine-tune later if you want)
for p in teacher_model.parameters():
    p.requires_grad = False

print("Teacher model ready (DINOv2 + DETR).")

# ============================================================
# 5. STUDENT MODEL: Swin Transformer + DETR HEAD
# ============================================================

# Swin backbone from timm – returns feature maps
swin_backbone = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    features_only=True,
    out_indices=[3]  # last stage features
).to(device)

for p in swin_backbone.parameters():
    p.requires_grad = True  # student backbone is trainable

# Get feature dimension (C) of last stage
dummy = torch.randn(1, 3, img_size, img_size).to(device)
with torch.no_grad():
    feats = swin_backbone(dummy)[0]  # [1, C, H, W]
student_backbone_dim = feats.shape[1]
print("Student backbone dim:", student_backbone_dim)


class SwinDETR(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images):
        """
        images: [B, 3, H, W] in [0,1]
        """
        feats = self.backbone(images)[0]          # [B, C, H, W]
        B, C, H, W = feats.shape
        memory = feats.flatten(2).transpose(1, 2)  # [B, S, C], S = H*W

        pred_logits, pred_boxes = self.head(memory)
        return pred_logits, pred_boxes


student_head = SimpleDETRHead(
    d_model=student_backbone_dim,
    nhead=8,
    num_queries=100,
    num_classes=len(VOC_CLASSES) + 1,
    num_decoder_layers=3
)

student_model = SwinDETR(swin_backbone, student_head).to(device)

print("Student model ready (Swin + DETR).")

# ============================================================
# 6. LLM TEXT EMBEDDINGS (DistilBERT) FOR CLASS ALIGNMENT
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
text_model.eval()
for p in text_model.parameters():
    p.requires_grad = False  # keep LLM frozen

# Create simple descriptions for each class
class_sentences = [f"A photo of a {c}." for c in VOC_CLASSES]

with torch.no_grad():
    tokens = tokenizer(
        class_sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    text_outputs = text_model(**tokens)
    # [num_classes, hidden_dim] using mean pooling
    text_embeds = text_outputs.last_hidden_state.mean(dim=1)  # (20, d_text)

text_embeds_dim = text_embeds.size(1)
print("Text embedding dim:", text_embeds_dim)

# Projection from text space -> student DETR feature space
text_proj = nn.Linear(text_embeds_dim, student_backbone_dim).to(device)

# ============================================================
# 7. DETR LOSS & MATCHING (REUSE / DEFINE HERE)
# ============================================================

from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w),
        (y_c - 0.5 * h),
        (x_c + 0.5 * w),
        (y_c + 0.5 * h),
    ]
    return torch.stack(b, dim=-1)


def hungarian_match(pred_logits, pred_boxes, targets):
    """
    For DETR style matching (used by both teacher & student).
    """
    B, Q, C = pred_logits.shape
    indices = []

    for b in range(B):
        out_prob = pred_logits[b].softmax(-1)        # [Q, C]
        out_bbox = pred_boxes[b]                     # [Q, 4]

        tgt_labels = targets[b]["labels"].to(pred_logits.device)
        tgt_boxes  = targets[b]["boxes"].to(pred_boxes.device)  # xyxy px

        if tgt_boxes.numel() == 0:
            indices.append(
                (torch.empty(0, dtype=torch.long),
                 torch.empty(0, dtype=torch.long))
            )
            continue

        tgt_boxes_norm = tgt_boxes / img_size  # [0,1]

        # classification cost
        cost_class = -out_prob[:, tgt_labels]  # [Q, N]

        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        cost_bbox = torch.cdist(out_bbox_xyxy, tgt_boxes_norm, p=1)   # [Q, N]
        cost_iou  = 1 - box_iou(out_bbox_xyxy, tgt_boxes_norm)        # [Q, N]

        C_cost = cost_class + 5.0 * cost_bbox + 2.0 * cost_iou
        C_cost_np = C_cost.detach().cpu().numpy()  # detach for scipy

        row_ind, col_ind = linear_sum_assignment(C_cost_np)
        indices.append(
            (torch.as_tensor(row_ind, dtype=torch.long),
             torch.as_tensor(col_ind, dtype=torch.long))
        )

    return indices


def detr_loss(pred_logits, pred_boxes, targets,
              lambda_cls=1.0, lambda_bbox=5.0, lambda_iou=2.0):
    indices = hungarian_match(pred_logits, pred_boxes, targets)

    B, Q, C = pred_logits.shape

    loss_cls = 0.0
    loss_bbox = 0.0
    loss_iou = 0.0
    num_tgt_total = 0

    for b in range(B):
        idx_pred, idx_tgt = indices[b]
        tgt = targets[b]

        if idx_tgt.numel() == 0:
            continue

        num_tgt_total += idx_tgt.numel()

        # classification loss
        src_logits = pred_logits[b, idx_pred]  # [N, C]
        tgt_labels = tgt["labels"][idx_tgt].to(pred_logits.device)
        loss_cls_b = Fnn.cross_entropy(src_logits, tgt_labels)
        loss_cls += loss_cls_b

        # bbox loss
        src_boxes = pred_boxes[b, idx_pred]  # cxcywh
        tgt_boxes = tgt["boxes"][idx_tgt].to(pred_boxes.device)  # xyxy px
        tgt_boxes_norm = tgt_boxes / img_size

        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        loss_bbox_b = Fnn.l1_loss(src_boxes_xyxy, tgt_boxes_norm)
        loss_bbox += loss_bbox_b

        # IoU loss
        iou_matrix = box_iou(src_boxes_xyxy, tgt_boxes_norm)
        diag_iou = iou_matrix[torch.arange(idx_tgt.numel()),
                              torch.arange(idx_tgt.numel())]
        loss_iou_b = (1 - diag_iou).mean()
        loss_iou += loss_iou_b

    if num_tgt_total == 0:
        return pred_logits.sum() * 0.0  # scalar zero

    loss = lambda_cls * loss_cls + lambda_bbox * loss_bbox + lambda_iou * loss_iou
    return loss

# ============================================================
# 8. KNOWLEDGE DISTILLATION + LLM LOSS
# ============================================================

T_kd = 2.0          # temperature for KD
alpha_kd_cls = 0.5  # weight for KD classification loss
beta_kd_box  = 1.0  # weight for KD box loss
gamma_llm    = 0.1  # weight for LLM/text alignment loss


def kd_losses(student_logits, student_boxes, teacher_logits, teacher_boxes):
    # ---- classification KD (KL divergence between softened logits) ----
    s_log_soft = Fnn.log_softmax(student_logits / T_kd, dim=-1)
    t_soft     = Fnn.softmax(teacher_logits / T_kd, dim=-1)

    kd_cls = Fnn.kl_div(
        s_log_soft,
        t_soft,
        reduction="batchmean"
    ) * (T_kd * T_kd)

    # ---- box KD (MSE between all predicted boxes) ----
    kd_box = Fnn.mse_loss(student_boxes, teacher_boxes)

    return kd_cls, kd_box


def llm_alignment_loss(student_head, text_embeds, text_proj):
    """
    Align student classifier weights with projected LLM text embeddings.
    """
    # ignore background class weight (last one)
    cls_weights = student_head.class_embed.weight[:len(VOC_CLASSES)]  # [20, d_model]
    text_proj_emb = text_proj(text_embeds.to(cls_weights.device))     # [20, d_model]

    return Fnn.mse_loss(cls_weights, text_proj_emb)

# ============================================================
# 9. TRAINING: STUDENT (Swin+DETR) WITH TEACHER + LLM
# ============================================================

num_epochs = 5
learning_rate = 1e-4

optimizer = optim.AdamW(
    list(student_model.parameters()) + list(text_proj.parameters()),
    lr=learning_rate
)

train_loss_history = []
val_loss_history   = []
val_map_history    = []


def evaluate_student(model, data_loader, max_batches=20):
    model.eval()
    total_loss = 0.0
    all_ious = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            pred_logits, pred_boxes = model(images)
            loss = detr_loss(pred_logits, pred_boxes, targets)
            total_loss += loss.item()

            indices = hungarian_match(pred_logits, pred_boxes, targets)
            for b in range(len(targets)):
                idx_pred, idx_tgt = indices[b]
                if idx_tgt.numel() == 0:
                    continue
                pb = box_cxcywh_to_xyxy(pred_boxes[b, idx_pred])   # [N,4]
                tb = targets[b]["boxes"][idx_tgt] / img_size       # [N,4]
                ious = box_iou(pb, tb)
                diag_iou = ious[torch.arange(idx_tgt.numel()),
                                torch.arange(idx_tgt.numel())]
                all_ious.extend(diag_iou.cpu().tolist())

    avg_loss = total_loss / max(1, min(max_batches, len(data_loader)))
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return avg_loss, mean_iou


for epoch in range(1, num_epochs + 1):
    student_model.train()
    running_loss = 0.0

    print(f"\n==== Epoch {epoch}/{num_epochs} ====")
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits, teacher_boxes = teacher_model(images)

        # Student predictions
        student_logits, student_boxes = student_model(images)

        # Detection loss (student vs ground truth)
        det_loss = detr_loss(student_logits, student_boxes, targets)

        # KD losses (student vs teacher)
        kd_cls_loss, kd_box_loss = kd_losses(
            student_logits, student_boxes,
            teacher_logits, teacher_boxes
        )

        # LLM alignment loss
        llm_loss = llm_alignment_loss(student_head, text_embeds, text_proj)

        # Total loss
        loss = det_loss + alpha_kd_cls * kd_cls_loss + beta_kd_box * kd_box_loss + gamma_llm * llm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 🔹 Print detailed loss per iteration
        print(
            f"Epoch {epoch} | Iter {i+1}/{len(train_loader)} | "
            f"Det: {det_loss.item():.4f} | KD_cls: {kd_cls_loss.item():.4f} | "
            f"KD_box: {kd_box_loss.item():.4f} | LLM: {llm_loss.item():.4f} | "
            f"Total: {loss.item():.4f}"
        )

    avg_train_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # 🔹 Evaluate student
    val_loss, val_map = evaluate_student(student_model, val_loader, max_batches=20)
    val_loss_history.append(val_loss)
    val_map_history.append(val_map)

    print(f"==> Epoch {epoch} Summary: "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val mIoU (proxy mAP): {val_map:.4f}")

# ============================================================
# 10. PLOTS: LOSS & METRICS
# ============================================================

epochs_range = range(1, num_epochs + 1)

plt.figure()
plt.plot(epochs_range, train_loss_history, label="Train Loss")
plt.plot(epochs_range, val_loss_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Student (Swin+DETR) Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs_range, val_map_history, label="Val mIoU (proxy mAP)")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Student (Swin+DETR) Validation Metric")
plt.legend()
plt.grid(True)
plt.show()