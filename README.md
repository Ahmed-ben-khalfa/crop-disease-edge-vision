# Crop Disease Edge Vision 🌾🔬

**A robust, field-ready Agricultural Foundation Model for real-time crop disease diagnosis on Edge devices.**

Most plant disease diagnostic models achieve 99% accuracy in the lab (e.g., PlantVillage) but collapse to 30-40% when tested in the real world due to the **Domain Gap** (complex backgrounds, shadows, multiple leaves). 

**Crop Disease Edge Vision** solves this by leveraging **DINOv2** (a vision foundation model) and advanced Out-of-Distribution (OOD) detection to create a highly robust, "in-the-wild" diagnostic tool that tells you what's wrong with your crops—and refuses to guess if you take a picture of a shoe.

---

## ✨ Key Features & Unprecedented Robustness

*   **Foundation Model Backbone:** Uses **DINOv2-small** (frozen) as a feature extractor. Its self-supervised attention mechanism naturally isolates the leaf from the background without needing a dedicated segmentation model.
*   **Domain Gap Bridged:** Trained on a rigorously audited mix of PlantVillage (lab) and PlantDoc (field) data. We discovered and fixed a major data leakage in the academic PlantDoc dataset (crops from the same image were leaking across splits) using strict `GroupShuffleSplit`.
*   **Scientific OOD Detection:** Uses **Mahalanobis Distance** to reject out-of-distribution images. If you upload a picture of a non-leaf object or a completely unknown species, the model will flag it instead of confidently predicting a random disease. 
*   **Adaptive Severity Estimation (TTA):** Computes the percentage of the diseased leaf area via HSV color thresholding. **It is adaptive:** if the model detects a species with naturally red/brown leaves (like Blueberry) or autumn senescence, it dynamically adjusts the "healthy" color range to avoid false positives. It uses **Test-Time Augmentation (TTA)** to stabilize variance.
*   **Multi-Disease Detection:** Flags co-infections when multiple disease probabilities cross critical thresholds.
*   **INT8 Quantization:** Optimized for Edge deployment. The model was successfully statically quantized to INT8, drastically reducing RAM footprint with negligible accuracy loss.

## 🚀 The Web Application

We provide a **Premium Glassmorphism Web App** built with FastAPI (Backend) and Vanilla HTML/CSS/JS (Frontend) replacing basic Gradio prototypes.

*   **Dark Mode & Fluid UI:** Professional agricultural dashboard.
*   **Real-time Alerts:** Warns against OOD images, multiple diseases, and diffuse spatial attention (non-leaf objects).
*   **Severity Progress Bar:** Visual indicator of the infection ratio.

### Quickstart

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install fastapi uvicorn python-multipart
    ```
2.  **Run the Web App:**
    ```bash
    python app/api.py
    ```
3.  Open your browser at `http://localhost:8000`

### CLI Usage
You can also run a quick diagnosis via the command line:
```bash
python src/predict.py path/to/your/leaf_image.jpg
```

---

## 🧠 Architecture Overview

1.  **Input:** RGB Image (in-the-wild).
2.  **Feature Extraction:** DINOv2-small (frozen). Output: 384-dimensional CLS embedding.
3.  **Classification Head:** Robust MLP (trained with blur, noise, brightness, and JPEG compression augmentations). Outputs probabilities for **38 classes** across **14 crops**.
4.  **OOD Pipeline:** Calculates Mahalanobis distance against 38 class prototypes. If distance > 99th percentile threshold, penalizes softmax confidence.
5.  **Severity Pipeline:** DINOv2 spatial attention mask isolates the leaf -> Adaptive HSV Thresholding -> Calculates lesion ratio.

## 📊 Performance 
*   **Lab Accuracy (PlantVillage):** ~98%
*   **Field Accuracy (PlantDoc Test - Cleaned):** 72.5% *(Massive improvement over generic baselines)*
*   **Robustness:** Maintained +15 points over standard models when subjected to severe sensor noise and motion blur.

---
*Built to bring true AI reliability to the agricultural edge.*
