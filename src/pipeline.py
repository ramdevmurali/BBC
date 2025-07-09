# src/pipeline.py

import torch
from tqdm import tqdm
from transformers import pipeline # Removed AutoModelForSequenceClassification, AutoTokenizer - they're not needed here
import functools # Used for caching models

from . import config

# --- Helper functions for cached pipeline initialization ---
# These functions ensure the pipeline objects are loaded only once.
# maxsize=1 effectively makes it a simple memoization (cache just one instance per unique call).

@functools.lru_cache(maxsize=1)
def _get_zero_shot_classifier(model_name, device):
    """Initializes and caches the zero-shot classifier pipeline."""
    print(f"Loading Zero-Shot Classifier: {model_name}...") # Prints only on first load
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device
    )

@functools.lru_cache(maxsize=1)
def _get_ner_pipeline(model_name, device):
    """Initializes and caches the NER pipeline."""
    print(f"Loading NER Model: {model_name}...") # Prints only on first load
    return pipeline(
        "ner",
        model=model_name,
        device=device,
        aggregation_strategy="simple"
    )

@functools.lru_cache(maxsize=1)
def _get_summarizer_pipeline(model_name, device):
    """Initializes and caches the summarization pipeline."""
    print(f"Loading Summarizer Model: {model_name}...") # Prints only on first load
    return pipeline(
        "summarization",
        model=model_name,
        device=device,
        # --- CRITICAL FIX: Explicitly handle long inputs for summarization ---
        # Models like BART have a max sequence length (often 1024).
        # We explicitly set truncation to prevent indexing errors for very long articles.
        max_length=1024,   
        truncation=True    
    )

@functools.lru_cache(maxsize=1)
def _get_finetuned_classifier(model_path, device):
    """Initializes and caches the fine-tuned classifier pipeline from a local path."""
    print(f"Loading Fine-Tuned Classifier from: {model_path}...") # Prints only on first load
    # For locally saved models, specify model and tokenizer explicitly if they're in the same folder
    return pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path, # Assumes tokenizer is saved with the model in the same directory
        device=device
    )

# --- Existing task functions, now utilizing the cached helpers ---

def classify_sub_categories(texts: list[str], candidate_labels: list[str]) -> list[str]:
    """
    Performs zero-shot classification on a list of texts using a cached pipeline.
    """
    # This print message indicates the start of this specific classification task (may be called multiple times)
    print(f"Starting sub-category classification with {len(candidate_labels)} labels...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = _get_zero_shot_classifier(config.ZERO_SHOT_CLASSIFIER_MODEL, device) # Uses cached helper

    predictions = []
    for text in tqdm(texts, desc="Classifying articles"):
        result = classifier(text, candidate_labels, multi_label=False)
        top_label = result['labels'][0]
        predictions.append(top_label)
        
    print("Classification complete.")
    return predictions

def extract_entities(text: str) -> list[dict]:
    """
    Performs Named Entity Recognition on a text to extract entities using a cached pipeline.
    """
    device = 0 if torch.cuda.is_available() else -1
    ner_pipeline = _get_ner_pipeline(config.NER_MODEL, device) # Uses cached helper

    entities = ner_pipeline(text)
    # Filter for 'PER' (Person) and 'ORG' (Organization) entities
    filtered_entities = [
        entity for entity in entities 
        if entity['entity_group'] in ['PER', 'ORG']
    ]

    return filtered_entities

def summarize_text(text: str) -> str:
    """
    Creates a summary of a given text using a cached pipeline.
    """
    device = 0 if torch.cuda.is_available() else -1
    summarizer = _get_summarizer_pipeline(config.SUMMARIZER_MODEL, device) # Uses cached helper

    # The max_length and min_length here apply to the *output* summary's token length.
    # Input truncation is handled by the pipeline's initialization.
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    
    return summary[0]['summary_text']

def classify_with_finetuned_model(texts: list[str]) -> list[str]:
    """
    Performs classification using a custom fine-tuned model from a local path, using a cached pipeline.
    """
    print("Starting classification with fine-tuned model...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = _get_finetuned_classifier(config.FINETUNED_CLASSIFIER_MODEL_PATH, device) # Uses cached helper
    
    results = classifier(texts)
    # Extracts the predicted label from the pipeline's output
    predictions = [result['label'] for result in results]
    
    print("Classification complete.")
    return predictions