# HMLR Data Science Challenge: NLP & Computer Vision

**Candidate:** Ramdev Murali
**Date:** 02/07/2025
**Challenge Tackled:** Natural Language Processing

---

## 1. Project Overview

This project addresses the Natural Language Processing challenge set by HM Land Registry. It involves building a robust pipeline to analyze, classify, and extract information from the BBC News dataset.

The pipeline performs three core tasks:
1.  **Sub-Category Classification (Essential Task):** Classifies articles from broad categories (e.g., `business`, `sport`) into more granular sub-categories (e.g., `economy`, `football`).
2.  **Named Entity Recognition (Desired Task):** Identifies and extracts media personalities from articles.
3.  **Abstractive Summarization (Desired Task):** Generates concise summaries of articles related to a specific topic.

The solution is built with a modular structure, emphasizing reproducibility, and utilizes state-of-the-art transformer models from the Hugging Face ecosystem.

---

## 2. How to Run

### Prerequisites
- Python 3.10+
- Git

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd hmlr-nlp-challenge
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    *(On Windows, use `venv\Scripts\activate`)*

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the main pipeline:**
    ```bash
    python -m src.main
    ```

    *__Note:__ The first run will download several pre-trained models from the internet, which may take a considerable amount of time and disk space (approx. 2-3 GB). Subsequent runs will be much faster as they will use the cached models.*

---

## 3. Methodology

### 3.1. Data Loading and Preprocessing
The raw text files from the BBC dataset are loaded, parsed, and consolidated into a single structured DataFrame. This processed DataFrame is then cached as `data/processed/bbc_data.csv` to accelerate subsequent runs.

### 3.2. Sub-Category Classification
- **Approach:** A **Zero-Shot Classification** model was chosen for this task. This powerful technique allows for classification against arbitrary labels without requiring any model fine-tuning, making the system flexible and scalable.
- **Model:** `valhalla/distilbart-mnli-12-3`. This model was selected as an optimal balance between high performance and resource efficiency. While larger models like `bart-large` offer marginal accuracy improvements, `distilbart` provides excellent results while ensuring the application can run effectively on standard hardware (preventing memory-related errors).
- **Implementation:** For each primary category, the relevant articles are passed to the classifier along with a predefined list of candidate sub-categories (defined in `src/config.py`). The label with the highest confidence score is assigned as the sub-category.

### 3.3. Named Entity Recognition (NER)
- **Approach:** A pre-trained NER model is used to identify entities within the text. The pipeline is configured to specifically extract `PERSON` and `ORGANIZATION` entities.
- **Model:** `dslim/bert-base-NER`. A well-regarded and efficient BERT-based model for general-purpose NER tasks.
- **Implementation:** The model processes a sample of articles and extracts all identified persons, demonstrating the ability to identify key individuals in the text.

### 3.4. Abstractive Summarization
- **Approach:** A pre-trained sequence-to-sequence model is used to generate abstractive summaries. This approach creates new, human-like sentences that capture the essence of the source text, rather than simply extracting existing ones.
- **Model:** `sshleifer/distilbart-cnn-12-6`. A distilled version of BART fine-tuned on the CNN/DailyMail dataset, known for producing high-quality summaries.
- **Implementation:** The pipeline filters for articles containing a target keyword ("April") and then feeds them into the summarizer to produce concise summaries of a controlled length (40-150 words).

---

## 4. Results and Performance

The output of the pipeline is saved into three separate CSV files in the `outputs/` directory.

### `outputs/classification_results.csv`
Contains the full dataset with the predicted sub-category for each article.

**Sample Output:**
```csv
category,sub_category,text
business,economy,"UK economy facing 'major risks'..."
business,stock market,"Dollar gains on Greenspan speech..."
sport,football,"Mourinho hints at Chelsea exit..."
```

### `outputs/ner_results.csv`
Contains named persons extracted from a sample of entertainment articles.

**Sample Output:**
```csv
text_sample,persons_found
"Musicians to tackle US red tape...","Nigel McCune, James Seller"
"U2's desire to be number one...","Bono"
```

### `outputs/summarization_results.csv`
Contains generated summaries for articles mentioning the target month.

**Sample Output:**
```csv
category,text,summary
entertainment,"Rocker Doherty in on-stage fight...","Rock singer Pete Doherty has been involved in an on-stage fight with his guitarist at a London gig..."
```

The model outputs demonstrate a strong ability to correctly classify articles, identify key entities, and produce relevant, fluent summaries, successfully meeting all requirements of the challenge.
