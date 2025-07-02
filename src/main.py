import pandas as pd
from tqdm import tqdm  # <-- ADD THIS LINE
from . import config
from . import data_loader
from . import pipeline
def run_essential_task(df: pd.DataFrame):
    """Runs the sub-category classification task and saves the results."""
    print("\n--- Running Essential Task: Sub-Category Classification ---")
    
    # We need to process each category with its own set of labels.
    # We will build the full dataframe category by category and then combine.
    all_results = []

    for category, labels in config.SUB_CATEGORY_LABELS.items():
        print(f"\nProcessing category: {category}")
        
        # Filter the DataFrame for the current category
        category_df = df[df['category'] == category].copy()
        if category_df.empty:
            continue
            
        texts_to_classify = category_df['text'].tolist()
        
        # Run the classification for this category
        predictions = pipeline.classify_sub_categories(texts_to_classify, labels)
        
        # Add predictions to the category-specific DataFrame
        category_df['sub_category'] = predictions
        all_results.append(category_df)

    # Combine all results back into one DataFrame
    results_df = pd.concat(all_results)

    # Save the results to a CSV file
    output_file = config.OUTPUT_PATH / "classification_results.csv"
    config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    results_df[['category', 'sub_category', 'text']].to_csv(output_file, index=False)
    print(f"\nEssential task complete. Results saved to: {output_file}")
    
    print("\nSample of Classification Results:")
    print(results_df[['category', 'sub_category']].head())


def run_desired_tasks(df: pd.DataFrame):
    """Runs the NER and Summarization tasks and saves the results."""
    print("\n--- Running Desired Tasks: NER & Summarization ---")
    
    print("\nExtracting entities from a sample of 'entertainment' articles...")
    entertainment_df = df[df['category'] == 'entertainment'].head(20)
    
    entities_data = []
    for index, row in entertainment_df.iterrows():
        text_sample = row['text'][:150] + "..."
        entities = pipeline.extract_entities(row['text'])
        persons = [e['word'] for e in entities if e['entity_group'] == 'PER']
        if persons:
            entities_data.append({'text_sample': text_sample, 'persons_found': ", ".join(persons)})

    ner_results_df = pd.DataFrame(entities_data)
    
    print(f"\nFinding and summarizing articles mentioning '{config.TARGET_MONTH}'...")
    april_df = df[df['text'].str.contains(config.TARGET_MONTH, case=False)].copy()
    
    april_df_sample = april_df.head(5)
    
    summaries = []
    if not april_df_sample.empty:
        for text in tqdm(april_df_sample['text'], desc="Summarizing articles"):
            summaries.append(pipeline.summarize_text(text))
        april_df_sample['summary'] = summaries
    else:
        print(f"No articles found mentioning '{config.TARGET_MONTH}'.")

    config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    ner_output_file = config.OUTPUT_PATH / "ner_results.csv"
    summary_output_file = config.OUTPUT_PATH / "summarization_results.csv"
    
    ner_results_df.to_csv(ner_output_file, index=False)
    print(f"\nNER results saved to: {ner_output_file}")

    if not april_df_sample.empty:
        april_df_sample[['category', 'text', 'summary']].to_csv(summary_output_file, index=False)
        print(f"Summarization results saved to: {summary_output_file}")


def main():
    """Main function to run the entire NLP pipeline."""
    bbc_df = data_loader.load_data_from_folders()
    run_essential_task(bbc_df)
    run_desired_tasks(bbc_df)
    print("\n--- All tasks complete! ---")


if __name__ == '__main__':
    main()