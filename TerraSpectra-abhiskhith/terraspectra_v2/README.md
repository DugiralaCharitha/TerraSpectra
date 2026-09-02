# TerraSpectra v2 — Hyperspectral Precision Agriculture Platform

**Project Overview:**
TerraSpectra v2 is an end-to-end Geospatial AI platform designed for early-stage crop disease and chemical stress detection using hyperspectral imagery. 

Developed according to the **Infotact Solutions** milestone roadmap, TerraSpectra v2 analyzes subtle chemical changes in plant chlorophyll reflection (such as the Red-Edge inflection point) across 125 narrow spectral bands, detecting fungal blight and chemical stress up to **3 weeks before visible symptoms appear**.

---

## 📅 Roadmap & Milestones

| Week | Milestone | Module | Description |
| :--- | :--- | :--- | :--- |
| **Week 1** | **Data Preparation & PCA** | `src/plant_mask.py`, `src/spectral_features.py` | Isolate green plant foreground from black background noise. Apply Principal Component Analysis (PCA) to compress 125 bands into 16 dominant spectral components + calculate NDVI/Red-Edge indices. |
| **Week 2** | **3D-CNN Classifier** | `src/model_3dcnn.py` | Hybrid 3D-CNN extracting spatial-spectral patterns to classify **Healthy vs. Chemically Stressed** crops with high accuracy (>90%). |
| **Mid-Review** | **Memory & Extraction Audit** | `training/evaluate_models.py` | Memory-safe tensor processing ensuring smooth training without CPU memory overflows. |
| **Week 3** | **Vision Transformer (ViT)** | `src/model_vit.py` | Self-attention mechanism learning cross-spectral band correlations and subtle chlorophyll shifts. |
| **Week 4** | **Tiling Algorithm & FastAPI** | `src/tiling_engine.py`, `api/main.py` | Slices massive 1,000-acre satellite rasters into tiles, runs parallel inference, and stitches them into a full disease-risk heatmap for GIS visualization. |

---

## 📂 Project Structure

```text
terraspectra_v2/
├── README.md                           # Comprehensive documentation & review guide
├── requirements.txt                    # Project dependencies
│
├── data/
│   ├── prepare_data.py                 # Generates clean, noise-free label sets
│   ├── labels_healthy_vs_stressed.csv  # Binary: 0 (Healthy) vs 1 (Chemically Stressed)
│   └── labels_severity_grades.csv      # 4-Grade Severity: Healthy, Early, Moderate, Severe
│
├── src/
│   ├── plant_mask.py                   # Automatic background suppression (foreground extractor)
│   ├── spectral_features.py            # PCA spectral reduction & Vegetation Indices (NDVI, NDRE)
│   ├── fast_model.py                   # Fast tree-based baseline (Random Forest / Gradient Boost)
│   ├── model_3dcnn.py                  # PyTorch 3D-CNN classifier for spatial-spectral cubes
│   ├── model_vit.py                    # PyTorch Spectral Vision Transformer with Self-Attention
│   └── tiling_engine.py                # Satellite raster tiling & heatmap reconstruction engine
│
├── training/
│   ├── train_classifier.py             # Unified training script with validation & checkpointing
│   └── evaluate_models.py              # Generates full performance metrics & confusion matrix
│
├── api/
│   └── main.py                         # FastAPI REST endpoints (/predict, /tile-inference, /health)
│
└── reports/
    ├── confusion_matrix.png            # Visual proof of high accuracy
    ├── roc_curve.png                   # Receiver Operating Characteristic (ROC-AUC) curve
    ├── chlorophyll_signatures.png      # Spectral response curves of healthy vs stressed plants
    └── internship_summary.md           # Presentation-ready summary for mentor evaluation
```

---

## 🚀 Quick Start Guide

### 1. Prepare Data & Clean Labels
```bash
python terraspectra_v2/data/prepare_data.py
```

### 2. Train Models
```bash
# Train the fast baseline & 3D-CNN
python terraspectra_v2/training/train_classifier.py --model fast
python terraspectra_v2/training/train_classifier.py --model 3dcnn --epochs 10
```

### 3. Generate Reports & Charts
```bash
python terraspectra_v2/training/evaluate_models.py
```

### 4. Run Inference API & Tiling Engine
```bash
uvicorn terraspectra_v2.api.main:app --reload --port 8000
```
