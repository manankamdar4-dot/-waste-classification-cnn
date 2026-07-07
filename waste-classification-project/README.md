# Waste Classification using Transfer Learning

An image classification project that automatically sorts waste into 6 categories (cardboard, glass, metal, paper, plastic, trash) using deep learning. Built as a demonstration of transfer learning, comparative model evaluation, and explainable AI — relevant to IoT-based smart waste sorting systems.

## Problem Statement

Manual waste sorting is slow, inconsistent, and doesn't scale well. This project explores whether a lightweight, camera-based AI system could automatically classify waste in real time — a capability relevant to smart bins and IoT-integrated waste management systems.

## Dataset

**TrashNet** — a public dataset of ~2,500 images across 6 classes: cardboard, glass, metal, paper, plastic, and general trash. Images were captured under natural lighting on a plain background.

Source: [garythung/trashnet](https://github.com/garythung/trashnet)

## Approach

We used **transfer learning** — starting from models pretrained on ImageNet (1M+ general images) and fine-tuning only the final classification layer on our 6 waste categories. This lets us train accurate models in minutes instead of needing to train from scratch on a small dataset.

Two architectures were trained and compared:

| Model         | Why it was chosen                                              |
|---------------|------------------------------------------------------------------|
| MobileNetV2   | Lightweight, fast inference — suited for edge/mobile deployment |
| ResNet18      | Deeper architecture, generally higher accuracy — used as a comparison baseline |

Data augmentation (random flips, rotations) was applied during training to improve generalization.

## Results

| Model       | Accuracy | Precision | Recall | F1-score |
|-------------|----------|-----------|--------|----------|
| MobileNetV2 | 81.5%    | 81.9%     | 81.5%  | 81.2%    |
| ResNet18    | 77.6%    | 78.6%     | 77.6%  | 77.6%    |

Confusion matrices for both models are included in this repo (`confusion_matrix_MobileNetV2.png`, `confusion_matrix_ResNet18.png`).

## Explainability (Grad-CAM)

To verify the models were learning meaningful visual features (not just memorizing images), we generated **Grad-CAM heatmaps**, which highlight the image regions each model focused on to make its prediction. Warmer colors (red/yellow) indicate higher attention.

See `gradcam_MobileNetV2.png` and `gradcam_ResNet18.png` for sample visualizations.

## Project Structure

```
├── waste_classification_project.ipynb   # Main notebook (run in Google Colab)
├── mobilenetv2_waste_classifier.pth     # Trained MobileNetV2 weights
├── resnet18_waste_classifier.pth        # Trained ResNet18 weights
├── model_comparison_results.csv         # Accuracy/precision/recall/F1 comparison
├── confusion_matrix_MobileNetV2.png
├── confusion_matrix_ResNet18.png
├── gradcam_MobileNetV2.png
├── gradcam_ResNet18.png
└── README.md
```

## How to Run

1. Open `waste_classification_project.ipynb` in [Google Colab](https://colab.research.google.com)
2. Go to Runtime → Change runtime type → select T4 GPU
3. Run all cells from top to bottom

## Future Work

- Convert the MobileNetV2 model to **TensorFlow Lite / ONNX** for deployment on edge devices (e.g., Raspberry Pi, ESP32-CAM), enabling real-time on-device classification without needing an internet connection.
- Expand the dataset with real-world field images (varied lighting, backgrounds, angles) to improve robustness beyond controlled lab-style photos.
- Integrate with an IoT camera setup for live smart-bin sorting.

## Tech Stack

Python, PyTorch, torchvision, scikit-learn, Grad-CAM, Google Colab (T4 GPU)
