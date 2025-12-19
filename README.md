#QAPR-Net

This repository provides the official implementation of QAPR-Net, a query-adaptive prototype refinement framework for 3D-guided few-shot image classification.

QAPR-Net aims to improve prototype robustness under severe view discrepancy and limited supervision by introducing query-adaptive multi-view aggregation and prototype refinement mechanisms.

Overview

Few-shot image classification under cross-view settings remains challenging, especially when support samples provide incomplete or biased visual information.
QAPR-Net addresses this issue by leveraging multi-view 3D structural cues and query-adaptive refinement, enabling more stable and discriminative category representations.

The framework consists of:

Query-adaptive multi-view prototype aggregation

Spatial–semantic prototype refinement

Delta-guided channel selection for compact representations

Code Structure
QAPR-Net/
├── Model/
│   ├── Backbone/          # backbone
│   ├── Head/              # Few-shot heads
│   └── Img_few_shot_prj.py
├── Dataloader/
│   ├── ModelNet40_split.py
│   ├── ShapeNet55.py
│   └── Toy4K.py
├── util/
├── main.py
└── README.md

Usage

The repository supports 3D-supported few-shot classification under episodic training.

Training and evaluation are handled through main.py with configurable options for:

Dataset

Backbone

Few-shot head

Episode configuration

Example (ModelNet40):

python main.py \
  --dataset ModelNet40 \
  --fs_head OursNet \
  --n_way 5 \
  --k_shot 1 \
  --data_path $your path$


Note: Dataset preparation and detailed configuration are intentionally not fully specified here.

Datasets

The framework is designed to work with multi-view 3D datasets such as:

ModelNet40

Toys4K

Support samples are generated from multi-view 3D projections, while query samples can be either synthetic or real images depending on experimental settings.

Notes

This codebase is primarily intended for research and academic use.

Certain implementation details (e.g. data preprocessing and evaluation protocols) are omitted for clarity.

For experimental settings used in the paper, please refer to the manuscript.

License

This project is released for research purposes only.
