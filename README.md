# Deep Learning Paper Analysis and Reproduction

This repository is part of the final project for analyzing and reproducing results from a deep learning research paper. The focus of this project is on feature selection in tabular deep learning, specifically implementing and using **Deep Lasso**, a gradient-based feature selection method.

## **Research Paper Overview**

The work is inspired by the research paper:
> "A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning" (NeurIPS 2023)

This paper highlights challenges in feature selection for tabular data, focusing on deep learning architectures. It proposes **Deep Lasso**, a novel input-gradient-based method for feature selection that outperforms traditional approaches in scenarios with noisy, redundant, or second-order features.

Key Contributions:
- Benchmarked feature selection methods for downstream neural networks.
- Proposed **Deep Lasso**, leveraging input gradients for robust feature selection.
- Evaluated the effectiveness of Deep Lasso on real-world datasets.

## **Project Goals**
1. Reproduce the feature selection method (Deep Lasso) from the paper.
2. Implement a pipeline to train models using selected features.
3. Evaluate the pipeline using metrics like **Mean Squared Error (MSE)**.

---

## **Repository Structure**

- `src/feature_selection`: Contains Deep Lasso implementation.
  - `deep_lasso.py`: Implements feature importance calculation and feature selection.
- `src/training`: Main training and evaluation pipeline.
  - `train_deep_model.py`: Training script with feature selection and model evaluation.
- `src/utils`: Utilities for dataset loading.
  - `data_loader.py`: Loads datasets like California Housing.
- `datasets`: Placeholder for dataset files.
- `results`: Placeholder for results and logs.

---

## **Setup Instructions**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/<your-username>/deep-learning-paper-analysis.git
cd deep-learning-paper-analysis
