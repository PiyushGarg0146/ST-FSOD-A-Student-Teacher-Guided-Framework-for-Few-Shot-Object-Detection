🧠 ST-FSOD: Student–Teacher Guided Multi-Modal Few-Shot Object Detection

DINOv2 + Deformable DETR + Vicuna LLM + Knowledge Distillation

🔍 Overview

ST-FSOD is a next-generation Few-Shot Object Detection (FSOD) framework that integrates:

Student–Teacher Knowledge Distillation

DINOv2 Vision Transformer Backbone

Deformable DETR Proposal Generator

Vicuna-7B Large Language Model Classifier

Real Pascal VOC 2012 Dataset

EMA Teacher Updates for Stability

Vision–Language Multi-Modal Fusion

This architecture is designed for low-data detection problems, delivering strong generalization to novel classes using semantic knowledge, visual priors, and temporal consistency.

✨ Key Features
🧑‍🏫 Student–Teacher Distillation

EMA teacher model provides stable pseudo-labels

Student receives task loss + KL distillation loss + feature-level alignment

Reduces overfitting in low-shot training

Boosts accuracy on novel categories

🔍 DINOv2 Visual Backbone

Pretrained dinov2_vits14

High-quality patch-level embeddings

Frozen backbone → efficient training

🧩 Deformable DETR Proposal Generator

Multi-scale deformable attention

Query-based bounding box regression

Robust localization even with limited samples

💬 Vicuna LLM Classifier (lmsys/vicuna-7b-v1.5)

Generates rich semantic embeddings for all classes

Aligns image features with language priors

Strong classification in low-shot conditions

🎯 FSOD Support

1-shot, 3-shot, 5-shot evaluation

Few-shot subset extraction

Full visualization suite
