QAPR-Net
This repository provides the official PyTorch implementation of QAPR-Net, a query-adaptive prototype refinement framework for 3D-guided few-shot image classification.

QAPR-Net aims to improve prototype robustness under severe view discrepancy and limited supervision by introducing query-adaptive multi-view aggregation and prototype refinement mechanisms.

Overview
Few-shot image classification under cross-view settings remains challenging, especially when support samples provide incomplete or biased visual information. QAPR-Net addresses this issue by leveraging multi-view 3D structural cues and query-adaptive refinement, enabling more stable and discriminative category representations.

Key Features
Query-adaptive Multi-view Prototype Aggregation: Dynamically integrates multi-view information based on query specific features.

Spatial–Semantic Prototype Refinement: Refines prototypes by considering both spatial structures and semantic consistency.

Delta-guided Channel Selection: Utilizes a Delta Gate mechanism for compact and efficient channel-wise feature representation.

Code Structure
Plaintext

QAPR-Net/
├── Model/
│   ├── Backbone/          # Feature extractors (e.g., ResNet)
│   ├── Head/              # Few-shot heads (QAPR-Net, AINet, etc.)
│   └── Img_few_shot_prj.py # Core model wrapper
├── Dataloader/            # Data loading and splitting logic
│   ├── ModelNet40_split.py # Loader for ModelNet40-LS
│   ├── ShapeNet55.py
│   └── Toy4K.py           # Loader for Toys4K dataset
├── util/                  # Utility functions (metrics, logging)
├── main.py                # Main script for training and evaluation
└── README.md
Usage
Requirements
Python 3.x

PyTorch & torchvision

NumPy, Pillow, tqdm, yaml

Training & Evaluation
The training and evaluation are handled through main.py using an episodic training strategy.

Example (ModelNet40):

Bash

python main.py \
  --dataset ModelNet40 \
  --fs_head OursNet \
  --n_way 5 \
  --k_shot 1 \
  --data_path /path/to/ModelNet40-LS
Note: Detailed configuration files and hyperparameter settings are intentionally not fully specified in this public repository to protect research integrity.

Datasets
The framework is designed to work with multi-view 3D datasets:

ModelNet40: 3D CAD models from 40 categories.

Toys4K: A large-scale 3D dataset for object recognition.

Support samples are generated from multi-view 3D projections (e.g., 12 or 14 views), while query samples consist of synthetic or real images with optional background augmentation using the DTD dataset.

Notes
This codebase is primarily intended for research and academic use.

Certain implementation details, such as specific data preprocessing pipelines and optimal loss weights (AIS/QPA), are omitted for clarity and to maintain the competitive advantage of the original research.

For the exact experimental settings and parameters used in the paper, please refer to the published manuscript.

License
This project is released for research purposes only.
