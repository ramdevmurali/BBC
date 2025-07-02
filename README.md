# HM Land Registry: A Critical Analysis of NLP Model Performance

**Candidate:** Ramdev Murali
**Date:** July 2, 2025

---

## 1. Executive Summary

This repository documents a systematic investigation into solving a real-world text classification problem, as specified by the HM Land Registry NLP Challenge. The project's mandate was not merely to achieve a high accuracy score, but to explore the trade-offs between modern NLP techniques, from rapid, flexible baselining to high-performance, specialized fine-tuning.

The project successfully delivers on all core requirements, culminating in a state-of-the-art model with **~100% accuracy** on the primary classification task. This multi-stage approach demonstrates a robust skillset in model selection, rapid prototyping, performance optimization, and professional software engineering practices.

---

## 2. Technical Stack & Engineering Principles

This project adheres to professional software engineering and MLOps principles.

-   **Languages & Libraries:** Python 3.10+, PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`, Scikit-learn, Pandas.
-   **Architecture:**
    -   **Application (`/src`):** Modular, decoupled logic with centralized configuration. Designed for stability and inference.
    -   **Experimentation (`/notebooks`):** An interactive Colab notebook for GPU-accelerated model training and exploratory data analysis.
-   **Reproducibility:** The environment is fully specified via `requirements.txt` and managed with `venv model in the main application (`src/main.py`).
-   **Deliverable:** The `outputs/classification_results.csv` file.

#### **✅ Desired Task 1: Named Entity Recognition**
-   **Requirement:** *"Identify documents and extract the named entities for media personalities, clearly identifying their jobs."*
-   **Status:** **Partially Fulfilled.**
-   **Implementation:** An NER model successfully extracts `PERSON` entities from the text. The more advanced sub-task of **Relation Extraction** (linking an entity to its role) was identified as a challenging component. In line with the project guidelines which 'encourage submission of solutions even if it only partially meets the requirements,' this was scoped out for this prototype.
-   **Deliverable:** The `outputs/ner_results.csv` file demonstrates the successful entity extraction.

#### **✅ Desired Task 2: Event Summarization**
-   **Requirement:** *"Extract summaries of anything that took place or is/was scheduled to take place in April."*
-   **Status:** **Fulfilled.**
-   **Implementation:** The main pipeline filters for articles containing "April" and uses a generative summarizer.
-   **Deliverable:** The `outputs/summarization_results.csv` file.

---

## 5. Architectural & Model Strategy

A "right tool for the job" philosophy was adopted, balancing performance with pragmatism.

-   **Application Structure:** The project is built on a modular architecture (`/src`) to ensure stability and maintainability. Experimentation and model training are separated into a `/notebooks` directory, mirroring professional MLOps workflows.

-   **Model Selection Strategy:**
    -   **For Classification:** To achieve maximum performance on the core task, a `DistilBERT` model was **fine-tuned** to create a hyper-specialized expert for this specific dataset.
    -   **For NER & Summarization:** To deliver these features efficiently, proven, **pre-specialized models** were leveraged. This included an off-the-shelf NER model (`dslim/bert-base-NER`) and a generative encoder-decoder model for summarization, which is architecturally suited for the task. This pragmatic approach avoids redundant training and demonstrates efficient use of existing state-of-the-art tools.

---

## 6. Performance Deep Dive: The Fine-Tuned Model

After establishing a 60% accuracy baseline with a Zero-Shot model, the fine-tuned `DistilBERT` achieved **~100% accuracy** on the unseen test set, effectively solving the classification task.

**Final Classification Report (Fine-Tuned Model):**
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

**Analysis of the ~100% Score:**
This near-perfect result is interpreted not as simple success, but as evidence of **hyper-specialization**. The model has mastered the specific linguistic patterns of the *2005 BBC News corpus*. While it has successfully generalized to the held-out test set from the *same distribution*, this mastery is considered "brittle."

![Final Confusion Matrix](outputs/finetuned_confusion_matrix.png)

---

## 7. Production Readiness & Next Steps

Based on this analysis, the fine-tuned model, despite its score, is not immediately production-ready. The critical next steps would be:

1.  **Test for Domain Shift:** Evaluate the model on out-of-distribution data (e.g., news from 2025 or from a different publisher) to measure its true real-world generalization.
2.  **Develop a Re-training Strategy:** Design a strategy for continuous monitoring and periodic re-training to ensure the model remains accurate as language and topics evolve.
3.  **Deploy as a Service:** Integrate the finalized model into the robust `/src` application, containerize it (e.g., with Docker), and deploy it as a scalable inference API.