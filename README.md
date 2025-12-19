# QAPR-Net

This repository provides the official implementation of **QAPR-Net**,  
Adaptive Multi-View Refinement for **Enhanced 3D-Guided Few-Shot Image Classification**.

---

## 1. Introduction

Few-shot image classification under cross-view settings remains challenging, especially when
support samples provide incomplete or biased visual information.
Existing approaches often rely on uniform multi-view aggregation, making them sensitive
to view discrepancies and background noise.

**QAPR-Net** addresses this problem by introducing *query-adaptive multi-view aggregation*
and *prototype refinement mechanisms*, enabling the model to construct more robust
and discriminative category representations under limited supervision.

---

## 2. Key Components

- **Query-adaptive Multi-view Prototype Aggregation**  
  Dynamically integrates multi-view information conditioned on query-specific features.

- **Spatial–Semantic Prototype Refinement**  
  Refines category prototypes by jointly modeling spatial structures and semantic consistency.

- **Delta-guided Channel Selection**  
  Employs a Delta Gate mechanism to select discriminative channels and suppress redundancy.

---

## 3. Code Structure
The repository is organized as follows:

```text
QAPR-Net/
├── dataset/
│   ├──ModelNet40-LS
│   ├──Toys4k              #e.t.c
├── Model/
│   ├── Backbone/          # Feature extractors
│   ├── Head/              # Few-shot heads
│   └── Img_few_shot_prj.py # Core 3D-support model wrapper
├── Dataloader/            # Data loading modules
│   ├── ModelNet40_split.py # ModelNet40
│   ├── Toy4K.py           # Toy4K
├── Pretrain/            # Data loading modules
│   ├── Data_Loader
│   ├── main_pretrain.py
├── util/                  # Utility functions
├── main.py                # Training and evaluation
└── README.md
