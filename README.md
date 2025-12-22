# QAPR-Net

This repository provides the official implementation of **QAPR-Net**,  
Adaptive Multi-View Refinement for **Enhanced 3D-Guided Few-Shot Image Classification**.

---

## 1. Introduction

**QAPR-Net** addresses this problem by introducing *query-adaptive multi-view aggregation*
and *prototype refinement mechanisms*, enabling the model to construct more robust
and discriminative category representations under limited supervision.

---

## 2. Code Structure
The repository is organized as follows:

```text
QAPR-Net/
├── dataset/
│   ├──ModelNet40-LS
│   ├──Toys4k              # etc.
├── Model/
│   ├── Backbone/          # Feature extractors
│   ├── Head/              # Few-shot heads
│   └── Img_few_shot_prj.py # Core 3D-support model wrapper
├── Dataloader/            # Data loading modules
│   ├── ModelNet40.py      # ModelNet40
│   ├── Toy4K.py           # Toy4K
├── Pretrain/            # Data loading modules
│   ├── Data_Loader
│   │   ├──ModelNet40.py    #etc.
│   ├── Pretrain_Loader
│   ├── main_pretrain.py
├── util/                  # Utility functions
├── main.py                # Training and evaluation
└── README.md
```
---
## 3. Training and Pretraining

Example (ModelNet40):
```bash
python main.py \
  --exp_name $Your Exp Name$ \
  --dataset $Dataset used for training$ \
  --data_path $/path/to/ModelNet40-LS$ \
  --fs_head $QAPR_Net$ \
  --backbone backbone network
```
```bash
python Pretrain\main_pretrain.py \
  --exp_name $Your Exp Name$ \
  --dataset $Dataset used for pretraining$ \
  --data_path $/path/to/ModelNet40-LS$ \
  --fs_head $QAPR_Net$ \
  --backbone backbone network
```
---
## 4. Datasets
The framework is designed to work with multi-view 3D datasets, leveraging 3D structural cues for robust prototype refinement:
* **ModelNet40**: Standard 3D CAD dataset.
* **Toys4K**: A large-scale 3D dataset featuring **fine-grained** object categories for challenging recognition tasks.
> **Note**: Support samples are generated from multi-view 3D projections (14 views), while query samples can be either synthetic or real images depending on specific experimental configurations.Full access will be granted upon the official publication of the manuscript.
---
## 5. Notes
* This codebase is primarily intended for research and academic use.
* For the exact experimental settings and hyper-parameters used in our paper, please refer to the original manuscript.
---
## 📜 License

This project is released for **research purposes only**. The implementation focuses on illustrating the core ideas of QAPR-Net rather than serving as a fully optimized or production-ready system.
