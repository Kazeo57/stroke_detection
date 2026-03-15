# stroke_detection
<p align="center">
  <img src="assets/banner_results.png" alt="Stroke Detection Banner" width="800">
</p>

---

Here is a clear and professional translation of your text into English:

Translation: AI-Driven Stroke Diagnosis
Every year, stroke affects more than 15 million people worldwide. Among these cases, approximately 5 million results in death, and another 5 million lead to permanent, severe disabilities. Ultimately, nearly 75% of stroke victims suffer grave consequences or pass away. Stroke occurs in two primary forms: ischemic (the most common) and hemorrhagic. Crucially, the earlier the diagnosis is made—especially within the first few hours—the higher the chances of recovery and the more effective the treatment becomes.

In reality, however, providing an early diagnosis is not so simple. Symptoms can be vague, vary from one patient to another, or even be outright misleading. Added to this are frequent hospital constraints such as lack of time, a shortage of qualified personnel, and limited equipment. All these factors make a rapid diagnosis difficult to obtain in many settings. This is where the core problem lies: currently available clinical tools do not guarantee a fast, standardized, and low-cost diagnosis, particularly for acute ischemic strokes. When diagnosis is delayed, the entire care chain slows down, and the patient's chances of recovery diminish.

In this context, Artificial Intelligence represents a relevant solution. Thanks to its automated analysis capabilities—specifically on medical imaging like brain CT scans—AI can help physicians make decisions faster, more objectively, and more consistently across different cases. However, for this technology to be truly useful in the field, it must be reliable, based on high-quality data, and its results must remain interpretable.

This project is built upon that vision. The objective is to design and evaluate an automated diagnostic support system for ischemic strokes using AI techniques. The idea is to detect suspicious areas related to ischemia directly from brain images, providing healthcare providers with concrete support to act quickly before damage becomes irreversible.
---
## 🛠 Technologies & Tools

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![LaTeX](https://img.shields.io/badge/latex-%23008080.svg?style=for-the-badge&logo=latex&logoColor=white)
---
"## 🔬 Methodology: The Hybrid Approach

Unlike standard segmentation, this project implements a **Hybrid U-Net + Chan-Vese approach**:

1.  **Deep Learning Backbone:** A U-Net architecture (evaluated with various encoders like EfficientNet) to capture global features and initial lesion localization.
2.  **Level-Set Integration:** Use of the Chan-Vese algorithm for active contour refinement, allowing the model to better delineate ischemic regions where boundaries are diffuse.
3.  **Domain Adaptation:** Implementation of specific training strategies and data augmentation to overcome the "small data" constraint typical in specialized medical imaging.

---

## 📊 Performance & Key Results

The implementation of domain adaptation and hybrid refinement led to a significant performance leap compared to baseline architectures.

### Metrics Comparison
| Method | Mean IoU | Sparse Categorical Accuracy |
| :--- | :---: | :---: |
| Baseline (Simple U-Net) | 0.0135 | 42.1% |
| **Hybrid Approach + Domain Adaptation** | **0.7781** | **94.8%** |

### Training Evolution & Visual Results
<p align="center">
  <img src="assets/loss_curve.png" alt="Training Loss Curve" width="400">
  <img src="assets/miou_evolution.png" alt="mIoU Graph" width="400">
</p>
---

##  1. Démo & Inference (Quick Start)

### Video Demo

<video width="480" controls>
  <source src="./assets/demo.mp4" type="video/mp4">
  Watch the video.
</video>

### Exécuter l'inférence localement
To test model on new CT scan Image (format `.jpeg` ou `.png`,`.jpg`) :

1. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   python3 image_processing.py


### 2. Roadmap 
# Train using the optimized hybrid configuration
python recipes/custom_train.py --arch efficientnet_unet --epochs 100 --hybrid_refinement

