import pandas as pd
import json

def load_vifactcheck_data():
    """
    Load 300 samples from vifactcheck dataset and save to raw_test.json
    """
    # Load the test split of vifactcheck dataset
    splits = {
        'train': 'data/train-00000-of-00001.parquet', 
        'dev': 'data/dev-00000-of-00001.parquet', 
        'test': 'data/test-00000-of-00001.parquet'
    }
    
    print("Loading dataset from Hugging Face...")
    df = pd.read_parquet("hf://datasets/tranthaihoa/vifactcheck/" + splits["test"])
    
    print(f"Dataset loaded with {len(df)} samples")
    
    # Take first 300 samples
    df_sample = df.head(300)
    
    # Prepare data in the required format
    output_data = []
    
    for _, row in df_sample.iterrows():
        sample = {
            "context": row["Context"],
            "claim": row["Statement"], 
            "evidence": row["Evidence"],
            "label": row["labels"]
        }
        output_data.append(sample)
    
    # Save to JSON file
    output_file = "raw_test.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully saved {len(output_data)} samples to {output_file}")
    
    # Print some statistics
    print(f"\nDataset statistics:")
    print(f"- Total samples: {len(output_data)}")
    
    # Count label distribution
    label_counts = {}
    for sample in output_data:
        label = sample["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"- Label distribution: {label_counts}")
    
    # Show sample
    print(f"\nSample data:")
    print(f"Claim: {output_data[0]['claim'][:100]}...")
    print(f"Context: {output_data[0]['context'][:100]}...")
    print(f"Evidence: {output_data[0]['evidence'][:100]}...")
    print(f"Label: {output_data[0]['label']}")

if __name__ == "__main__":
    load_vifactcheck_data() 