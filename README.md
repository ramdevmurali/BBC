# HM Land Registry: NLP Data Science Challenge

**Candidate:** Ramdev Murali
**Date:** July 2, 2025

---

## 1. Executive Summary

This repository contains a comprehensive, end-to-end solution for the HM Land Registry's NLP Data Science Challenge. The project demonstrates a full data science lifecycle, from establishing a rapid baseline to developing and evaluating a state-of-the-art, high-performance model.

The final solution consists of two key components:
1.  **A Professional Application Prototype:** A well-structured Python application that uses an efficient Zero-Shot model to perform sub-category classification, named entity recognition, and summarization. This represents the baseline model.
2.  **A High-Performance Fine-Tuned Model:** A `DistilBERT` model, fine-tuned on the BBC dataset using a Google Colab GPU, which achieves **~100% accuracy** on the primary classification task.

This multi-stage approach showcases a robust skillset in model selection, rapid prototyping, performance optimization, and professional software engineering practices.

---

## 2. Tech Stack & Key Concepts

-   **Language:** Python 3.10+
-   **Core Libraries:** `PyTorch`, `Hugging Face Transformers`, `Hugging Face Datasets`, `Scikit-learn`, `Pandas`
-   **Key Concepts Demonstrated:** Transfer Learning, Zero-Shot Classification, Fine-Tuning, Model Evaluation, MLOps (Reproducible Environments, Version Control).
-   **Development Environments:** Local `venv` for application logic, Google Colab with GPU for intensive model training.

---

## 3. 🚀 Project Structure & Usage

This repository is structured to separate application code from experimental/training code.

-   `/src`: Contains the modular, professional application code for running inference with the models.
-   `/notebooks`: Contains the Jupyter/Colab notebooks used for experimentation, including the final `3_Fine_Tuning_DistilBERT.ipynb` used to train the high-performance model.
-   `/outputs`: Contains the final evaluation artifacts, such as the confusion matrix.
-   `/models`: (Local Only, Ignored by Git) This folder is the destination for the trained model artifacts downloaded from the Colab notebook.

### How to Run the Baseline Application

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ramdevmurali/HMLR.git
    cd HMLR
    ```

2.  **Set Up the Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Baseline Pipeline:**
    This command uses the Zero-Shot model for all tasks.
    ```bash
    python -m src.main
    ```

---

## 4. 🧠 Model Performance & Evolution

A two-stage methodology was employed to solve the classification task.

### Stage 1: Zero-Shot Baseline Model
Initially, a `valhalla/distilbart-mnli-12-3` Zero-Shot model was used. This approach offered maximum flexibility and established a solid performance baseline of **60% accuracy** without any specific training. This confirmed the viability of using general-purpose models for the task.

### Stage 2: Fine-Tuned State-of-the-Art Model
To achieve maximum performance, a `distilbert-base-uncased` model was fine-tuned specifically on the BBC dataset classification task. The training was performed in a Google Colab notebook to leverage GPU acceleration, reducing training time from hours to minutes.

This specialized training resulted in a state-of-the-art model that has effectively "solved" this classification task.

#### Final Performance Report (Fine-Tuned Model)
The fine-tuned model achieves **~100% accuracy** on the unseen test set, demonstrating the immense power of transfer learning.

```
               precision    recall  f1-score   support

     business       1.00      1.00      1.00        51
entertainment       0.97      1.00      0.99        39
     politics       1.00      0.98      0.99        42
        sport       1.00      1.00      1.00        51
         tech       1.00      1.00      1.00        40

     accuracy                           1.00       223
    macro avg       0.99      1.00      1.00       223
 weighted avg       1.00      1.00      1.00       223
```

#### Final Confusion Matrix
The confusion matrix is nearly perfect, with almost all predictions falling on the main diagonal. This confirms the model's ability to distinguish between categories with extremely high precision and recall.

![Confusion Matrix](outputs/finetuned_confusion_matrix.png)