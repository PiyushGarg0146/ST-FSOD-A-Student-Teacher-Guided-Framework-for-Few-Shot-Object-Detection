# 🧠 ST-FSOD: Student–Teacher Guided Multi-Modal Few-Shot Object Detection

**Combining Vision Transformers, Language Models, and Knowledge Distillation for Few-Shot Object Detection**

---

## 🔍 Overview

**ST-FSOD** is a next-generation **Few-Shot Object Detection (FSOD)** framework that unifies visual, linguistic, and distillation-based learning.  
It integrates:

- 🧑‍🏫 **Student–Teacher Knowledge Distillation**
- 🧠 **DINOv2 Vision Transformer Backbone**
- 🔩 **Deformable DETR Proposal Generator**
- 💬 **Vicuna-7B LLM Classifier**
- 🧮 **EMA Teacher Updates for Stability**
- 🎨 **Vision–Language Multi-Modal Fusion**

Built for **low-data detection**, this architecture achieves **robust generalization to novel categories** using semantic priors, visual embeddings, and temporal consistency.

---

## ✨ Key Features

### 🧑‍🏫 Student–Teacher Distillation
- EMA (Exponential Moving Average) **teacher model** provides stable pseudo-labels.  
- **Student model** receives:
  - Task loss (detection loss)
  - KL divergence (distillation loss)
  - Feature-level alignment loss  
- Reduces overfitting in **few-shot** training.  
- Improves accuracy on **novel classes**.

---

### 🔍 DINOv2 Visual Backbone
- **Pretrained `dinov2_vits14`** model.  
- High-quality **patch-level embeddings**.  
- Backbone **frozen during training** → stable + efficient learning.

---

### 🧩 Deformable DETR Proposal Generator
- Multi-scale **deformable attention** for spatial precision.  
- Query-based **bounding box regression**.  
- Robust **localization** even with very few examples.

---

### 💬 Vicuna LLM Classifier (`lmsys/vicuna-7b-v1.5`)
- Generates **semantic embeddings** for each class.  
- Aligns **visual features** (from DINOv2) with **language priors**.  
- Enables **semantic transfer** → better performance in **low-shot** conditions.

---

## 🎯 FSOD Capabilities
- Supports **1-shot, 3-shot, and 5-shot** learning setups.  
- Built-in **few-shot subset extraction** and **evaluation suite**.  
- Integrated **visualization tools** for detection outputs and class attention.

---

## 🧱 Architecture Diagram

