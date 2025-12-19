# QAPR-Net

This repository provides the official PyTorch implementation of **QAPR-Net**,  
a query-adaptive prototype refinement framework for **3D-guided few-shot image classification**.

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

