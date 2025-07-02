import torch
from tqdm import tqdm
from transformers import pipeline
from . import config

def classify_sub_categories(texts: list[str], candidate_labels: list[str]) -> list[str]:
    """Performs zero-shot classification on a list of texts."""
    print(f"Starting sub-category classification with {len(candidate_labels)} labels...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model=config.ZERO_SHOT_CLASSIFIER_MODEL,
        device=device
    )
    predictions = []
    for text in tqdm(texts, desc="Classifying articles"):
        result = classifier(text, candidate_labels, multi_label=False)
        top_label = result['labels'][0]
        predictions.append(top_label)
    print("Classification complete.")
    return predictions

def extract_entities(text: str) -> list[dict]:
    """Performs Named Entity Recognition on a text to extract entities."""
    device = 0 if torch.cuda.is_available() else -1
    ner_pipeline = pipeline(
        "ner",
        model=config.NER_MODEL,
        device=device,
        aggregation_strategy="simple"
    )
    entities = ner_pipeline(text)
    filtered_entities = [
        entity for entity in entities if entity['entity_group'] in ['PER', 'ORG']
    ]
    return filtered_entities

def summarize_text(text: str) -> str:
    """Creates a summary of a given text."""
    # Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    
    # Initialize the summarization pipeline
    summarizer = pipeline(
        "summarization",
        model=config.SUMMARIZER_MODEL,
        device=device
    )

    # --- ADD THIS TRUNCATION LOGIC ---
    # The model can only handle 1024 tokens. We'll truncate the text to be safe.
    # We split by space and take the first 800 words as a rough proxy.
    truncated_text = " ".join(text.split()[:800])
    # ---------------------------------

    # Generate the summary USING THE TRUNCATED TEXT
    summary = summarizer(truncated_text, max_length=150, min_length=40, do_sample=False)
    
    # The result is a list with one dictionary, we extract the summary text.
    return summary[0]['summary_text']