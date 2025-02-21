import json
import sys
import os
from typing import List, Dict
import difflib
from datasets import load_dataset
import random
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Add build directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/build/src
build_dir = os.path.dirname(current_dir)  # /path/to/build
sys.path.insert(0, build_dir)

from utils.diff_lib import get_diff_ids, split_into_words

def extract_hallucination_spans(rejected: str, chosen: str) -> List[Dict[str, int]]:
    """Extract spans of hallucinated text from rejected text"""
    # Split text into words
    rejected_words = split_into_words(rejected)
    chosen_words = split_into_words(chosen)
    
    # Use difflib to get differences
    matcher = difflib.SequenceMatcher(None, rejected_words, chosen_words)
    rejected_diff_indices = set(range(len(rejected_words)))
    
    # Remove matched parts
    for match in matcher.get_matching_blocks():
        for i in range(match.size):
            if match.a + i in rejected_diff_indices:
                rejected_diff_indices.remove(match.a + i)
    
    rejected_diff_ids = sorted(list(rejected_diff_indices))
    
    # Combine consecutive indices into spans
    spans = []
    if not rejected_diff_ids:
        return spans
    
    # Initialize the first span
    current_span_start = rejected_diff_ids[0]
    current_words = []
    
    # Iterate through all difference indices
    for i, idx in enumerate(rejected_diff_ids):
        current_words.append(rejected_words[idx])
        
        # If it's the last index or the next index is not consecutive
        if i == len(rejected_diff_ids) - 1 or rejected_diff_ids[i + 1] != idx + 1:
            # Save the current span
            spans.append({
                "start": current_span_start,
                "end": idx + 1,
                "text": " ".join(current_words)
            })
            # If not the last index, start a new span
            if i < len(rejected_diff_ids) - 1:
                current_span_start = rejected_diff_ids[i + 1]
                current_words = []
    
    return spans

def add_hallucination_tags(text: str, spans: List[Dict[str, int]]) -> str:
    """Add hallucination tags to text"""
    words = split_into_words(text)
    tagged_words = words.copy()
    
    # Add tags from back to front to avoid index changes
    for span in reversed(spans):
        start, end = span["start"], span["end"]
        tagged_words[start:end] = [f"<hallucination>{' '.join(words[start:end])}</hallucination>"]
    
    return " ".join(tagged_words)

def process_sample(sample: Dict) -> Dict:
    """Process a single data sample"""
    text_data = json.loads(sample["text"])
    rejected = text_data["rejected"]
    chosen = text_data["chosen"]
    prompt = text_data.get("question", "What do you observe in this image?")  # Add default prompt
    
    # Extract hallucination spans
    hallucination_spans = extract_hallucination_spans(rejected, chosen)
    
    # Add tags
    tagged_text = add_hallucination_tags(rejected, hallucination_spans)
    
    return {
        "id": sample.get("idx", "unknown"),  # Get id from the original dataset's idx field
        "image_path": sample["image_path"],
        "original_solution": chosen,
        "hallucinated_solution": tagged_text,
        "test_solution": rejected,
        "prompt": prompt  # Add prompt field
    }

def process_dataset(num_samples: int = 500):
    """Process RLHF-V dataset
    
    Args:
        num_samples (int): Number of data entries to generate, default is 500
    """
    print("Starting to load RLHF-V dataset...")
    dataset = load_dataset("HaoyeZhang/RLHF-V-Dataset")
    train_data = list(dataset['train'])

    # Only take the first num_samples data
    train_data = train_data[:num_samples]
    
    # Ensure the image output directory exists
    image_output_dir = "evaluate/images/rlhfv"
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Clear the image output directory
    for filename in os.listdir(image_output_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(image_output_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")
    
    processed_data = []
    failed_samples = []
    pbar = tqdm(total=num_samples, desc="Processing RLHF-V dataset")
    
    for sample in train_data:
        try:
            processed_sample = process_sample(sample)
            
            # Process image
            if sample.get('image_path'):
                src_image_path = os.path.join('build/images/rlhfv', sample['image_path'])
                dst_image_path = os.path.join(image_output_dir, os.path.basename(sample['image_path']))
                
                if os.path.exists(src_image_path):
                    os.system(f'cp "{src_image_path}" "{dst_image_path}"')
                    # Only save the filename
                    processed_sample['image_path'] = os.path.basename(sample['image_path'])
                else:
                    raise FileNotFoundError(f"Image file does not exist: {src_image_path}")
            
            processed_data.append(processed_sample)
            pbar.update(1)
            
        except Exception as e:
            failed_samples.append({
                "sample_id": sample.get("idx", "unknown"),
                "error": str(e)
            })
            print(f"\nError processing sample {sample.get('idx', 'unknown')}: {str(e)}")
            continue
    
    pbar.close()
    
    # Output processing statistics
    print(f"\nSuccessfully processed samples: {len(processed_data)}")
    print(f"Failed samples: {len(failed_samples)}")
    
    # Modify output path
    output_path = "evaluate/data/processed_rlhfv_dataset.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # If there are failed samples, save failure records
    if failed_samples:
        failed_samples_path = "evaluate/data/failed_rlhfv_samples.json"
        with open(failed_samples_path, "w", encoding="utf-8") as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        print(f"Failed sample information has been saved to: {failed_samples_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process RLHF-V dataset')
    parser.add_argument('--num_samples', type=int, default=2000,
                      help='Number of data entries to generate (default 2000)')
    
    args = parser.parse_args()
    process_dataset(args.num_samples) 