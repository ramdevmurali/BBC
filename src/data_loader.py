import pandas as pd
import os
from . import config

def load_data_from_folders() -> pd.DataFrame:
    """
    Loads raw text files from categorized folders, combines them into a
    single pandas DataFrame, and saves it to the processed data path.

    If the processed file already exists, it loads it directly to save time.
    """
    config.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    if config.PROCESSED_DATA_FILE.exists():
        print(f"Loading pre-processed data from {config.PROCESSED_DATA_FILE}")
        return pd.read_csv(config.PROCESSED_DATA_FILE)

    print("Processing raw data from scratch...")
    data = []
    
    for category in os.listdir(config.RAW_DATA_PATH):
        category_path = os.path.join(config.RAW_DATA_PATH, category)
        
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_path, filename)
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            text = f.read()
                            data.append({'category': category, 'text': text})
                    except Exception as e:
                        print(f"Could not read file {file_path}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(config.PROCESSED_DATA_FILE, index=False)
    print(f"Processed data saved to {config.PROCESSED_DATA_FILE}")
    
    return df