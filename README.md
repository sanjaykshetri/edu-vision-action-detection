# 🎓 edu-vision-action-detection  
*A PyTorch–powered computer-vision model for detecting observable classroom actions*

---

## 📌 Overview

This repository contains a modular deep-learning pipeline that detects **observable student actions** from classroom images — such as:

- ✍️ writing
- 🧑‍🏫 looking at board / attention-direction
- ✋ hand raised
- 📱 device visible (phone / laptop)

**No mental-state prediction** is performed — only **objective, physically observable actions.**

The project is part of a long-term roadmap toward building data-driven educational tools and **will later integrate as a micro-service** into a larger “Teacher Assistant Dashboard” ecosystem.

---

## 🚀 What This Repo Demonstrates (Portfolio Highlights)

| Skill Area | Evidence |
|------------|----------|
| Deep Learning | ResNet-based classifier, PyTorch training pipeline |
| Computer Vision | ImageFolder dataset, transforms, augmentation |
| Engineering Maturity | Modular repo, separation of model vs. UI, scalable architecture |
| Deployment-readiness | Streamlit inference UI planned, save/load model weights |

If you're reviewing this repo as a hiring manager:
> This project showcases end-to-end ML capability: data → model → training → deployment.

---

## 🧠 Project Architecture

![Architecture Diagram](./diagrams/architecture.png)

- **Modular Components:** Clear separation of data, model, and UI code
- **Scalability:** Easily add classes, augmentations, or models
- **Reproducibility:** Config files and scripts for every step

---

## 📂 Dataset Construction

- **Classes:** `hand_raised`, `writing`, `looking_board`, `device_use`
- **Data sources:** Royalty-free stock images (Pexels API), staged photos (optional)
- **Folder structure:**
  - `data/train/<class>/` — training images
  - `data/val/<class>/` — validation images
- **Automated scripts:**
  - `download_stock.py` — Download images for each class using Pexels API
  - `split_train_val.py` — Split dataset into 80% train, 20% val
- **Augmentation:**
  - In-pipeline: Random flip, rotation, color jitter, crop, blur

## 🏋️ Model Training

- **Framework:** PyTorch
- **Model:** ResNet18 (transfer learning)
- **Augmentation:** Advanced transforms in training pipeline
- **Training features:**
  - AdamW optimizer, dropout, early stopping
  - Metrics logging (loss, accuracy per epoch)
  - Confusion matrix visualization
  - Model checkpointing (best model saved)
- **Script:** `train.py`

## 📊 Evaluation & Visualization

- **Metrics:** Training loss, validation accuracy (plotted via `visualize_metrics.py`)
- **Confusion matrix:** Auto-saved for best model
- **Inference:**
  - `inference.py` — Predict class for any image

## 🌐 Streamlit Demo App

- **File:** `app.py`
- **Features:**
  - Upload an image or select from test dataset
  - See model prediction instantly
  - User-friendly interface for demo/portfolio

## 📝 How to Run

1. **Install dependencies:**
   ```
   pip install torch torchvision matplotlib pillow scikit-learn streamlit
   ```
2. **Download and split dataset:**
   ```
   python download_stock.py --class all --limit 100
   python split_train_val.py
   ```
3. **Train the model:**
   ```
   python train.py
   ```
4. **Visualize metrics:**
   ```
   python visualize_metrics.py
   ```
5. **Run Streamlit app:**
   ```
   streamlit run app.py
   ```

## 📸 Example Results

- Add screenshots of Streamlit app and confusion matrix here.

## 📚 Credits & License

- Images: Pexels, Unsplash, Pixabay (see scripts for details)
- Code: MIT License

---

*For questions or collaboration, open an issue or pull request!*

