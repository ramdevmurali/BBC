# src/evaluate.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from . import data_loader
from . import pipeline

def evaluate_model_performance():
    """
    Performs a quantitative evaluation of the zero-shot model on the
    main categories as a proxy task.
    """
    print("--- Starting Model Performance Evaluation (Proxy Task) ---")

    # 1. Load data
    df = data_loader.load_data_from_folders()
    
    # For a robust evaluation, we only use a fraction of the data to simulate a real test set
    # This also makes the evaluation run faster. Let's use 25% of the data.
    _, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['category'])

    print(f"Evaluation will run on a test set of {len(test_df)} articles.")

    # 2. Get true labels and texts from the test set
    true_labels = test_df['category'].tolist()
    texts_to_classify = test_df['text'].tolist()
    
    # 3. Get predictions from the model
    # The candidate labels are the main categories themselves.
    candidate_labels = df['category'].unique().tolist()
    
    predicted_labels = pipeline.classify_sub_categories(texts_to_classify, candidate_labels)

    # 4. Generate and print the classification report
    print("\n--- Classification Report ---")
    report = classification_report(true_labels, predicted_labels)
    print(report)

    # 5. Generate and save the confusion matrix
    print("\n--- Generating Confusion Matrix ---")
    cm = confusion_matrix(true_labels, predicted_labels, labels=candidate_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=candidate_labels, yticklabels=candidate_labels)
    plt.title('Confusion Matrix: Predicted vs. True Categories')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save the plot to the outputs folder
    output_path = config.OUTPUT_PATH / "confusion_matrix.png"
    config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    
    print(f"\nConfusion matrix saved to: {output_path}")
    print("--- Evaluation Complete ---")

if __name__ == '__main__':
    evaluate_model_performance()