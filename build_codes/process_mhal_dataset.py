import sys
import os
import json
from typing import List, Dict
import requests
import random  # 用于随机抽样
from tqdm import tqdm
import subprocess
import base64

# 添加build目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/build/src
build_dir = os.path.dirname(current_dir)  # /path/to/build
sys.path.insert(0, build_dir)

from utils.diff_lib import get_diff_ids, split_into_words
from utils.text_utils import get_word_spans

def encode_image(image_path: str) -> str:
    """Convert image to base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def download_coco_val2014_images(image_dir: str, image_ids: List[str]):
    """Download COCO 2014 validation images
    
    Args:
        image_dir: Directory to save images
        image_ids: List of image IDs to download
    """
    os.makedirs(image_dir, exist_ok=True)
    
    # Base URL for COCO 2014 validation set
    base_url = "http://images.cocodataset.org/val2014/"
    
    print(f"\nStarting to download COCO validation images...")
    successful_downloads = 0
    failed_downloads = []
    
    for img_path in tqdm(image_ids, desc="Downloading images"):
        if not img_path:  # Skip empty paths
            continue
            
        img_filename = os.path.basename(img_path)
        save_path = os.path.join(image_dir, img_filename)
        
        # Skip download if file already exists
        if os.path.exists(save_path):
            successful_downloads += 1
            continue
            
        # Construct full URL
        url = base_url + img_filename
        
        try:
            # Use curl to download the image, set timeout and retries
            cmd = f'curl -L --retry 3 --retry-delay 2 --connect-timeout 10 -o "{save_path}" "{url}"'
            result = subprocess.run(cmd, shell=True, capture_output=True)
            
            if result.returncode == 0 and os.path.exists(save_path):
                successful_downloads += 1
            else:
                failed_downloads.append(img_path)
                if os.path.exists(save_path):
                    os.remove(save_path)  # Remove possibly incomplete file
                    
        except Exception as e:
            print(f"Error downloading image {img_filename}: {str(e)}")
            failed_downloads.append(img_path)
            if os.path.exists(save_path):
                os.remove(save_path)
    
    print(f"\nImage download completed:")
    print(f"Successfully downloaded: {successful_downloads} images")
    print(f"Download failed: {len(failed_downloads)} images")
    if failed_downloads:
        print("\nThe following images failed to download:")
        for img in failed_downloads[:10]:  # Only show the first 10
            print(f"- {img}")
        if len(failed_downloads) > 10:
            print(f"... and {len(failed_downloads) - 10} more images not displayed")

def extract_hallucination_spans_mhal(original: str, corrected: str) -> List[Dict[str, int]]:
    """Extract start and end positions of hallucinated text in mhal-detect dataset"""
    # Get difference IDs
    original_diff_ids, _ = get_diff_ids(original, corrected)
    
    if not original_diff_ids:
        return []
    
    # Split text into words to build spans
    original_words = split_into_words(original)
    
    # Process difference IDs in order, merging consecutive differences
    spans = []
    start = original_diff_ids[0]
    current_end = start + 1
    
    for i in range(1, len(original_diff_ids)):
        current_id = original_diff_ids[i]
        # Only merge if difference IDs are consecutive
        if current_id == current_end:
            current_end += 1
        else:
            # Add current span
            if current_end > start:
                spans.append({
                    "start": start,
                    "end": current_end,
                    "text": " ".join(original_words[start:current_end])
                })
            # Start new span
            start = current_id
            current_end = start + 1
    
    # Add the last span
    if current_end > start and start < len(original_words):
        spans.append({
            "start": start,
            "end": current_end,
            "text": " ".join(original_words[start:current_end])
        })
    
    return spans

def add_hallucination_tags_mhal(text: str, spans: List[Dict[str, int]]) -> str:
    """Add hallucination tags to text in mhal-detect dataset"""
    if not spans:
        return text
        
    words = split_into_words(text)
    tagged_text = ""
    last_end = 0
    
    # Merge overlapping or adjacent spans
    merged_spans = []
    sorted_spans = sorted(spans, key=lambda x: x["start"])
    current_span = sorted_spans[0]
    
    for next_span in sorted_spans[1:]:
        # Only merge if spans are adjacent or overlapping
        if next_span["start"] <= current_span["end"]:
            current_span = {
                "start": current_span["start"],
                "end": max(current_span["end"], next_span["end"]),
                "text": " ".join(words[current_span["start"]:max(current_span["end"], next_span["end"])])
            }
        else:
            merged_spans.append(current_span)
            current_span = next_span
    merged_spans.append(current_span)
    
    # Use merged spans to add tags
    for span in merged_spans:
        # Add normal text before the span
        if span["start"] > last_end:
            tagged_text += " ".join(words[last_end:span["start"]]) + " "
        
        # Add tagged text
        hallucinated_text = " ".join(words[span["start"]:span["end"]])
        if hallucinated_text.strip():  # Only add tag if text is not empty
            tagged_text += f"<hallucination>{hallucinated_text}</hallucination> "
        
        last_end = span["end"]
    
    # Add remaining normal text
    if last_end < len(words):
        tagged_text += " ".join(words[last_end:])
    
    return tagged_text.strip()

def process_mhal_sample(sample: Dict, index: int) -> Dict:
    """Process a single sample from mhal-detect dataset"""
    original = sample["response"]
    annotations = sample["annotations"]
    prompt = sample["question"]
    
    # Extract hallucination spans (only using INACCURATE label)
    char_spans = []
    for ann in annotations:
        if ann["label"] == "INACCURATE":
            # Ensure span text matches original text
            span_text = original[ann["start"]:ann["end"]]
            if span_text == ann["text"]:
                char_spans.append({
                    "start": ann["start"],
                    "end": ann["end"],
                    "text": span_text
                })
            else:
                print(f"Warning: span text does not match:\nExpected: {ann['text']}\nActual: {span_text}")
    
    # Sort by start position
    char_spans.sort(key=lambda x: x["start"])
    
    # Merge overlapping spans
    merged_char_spans = []
    if char_spans:
        current_span = char_spans[0]
        for next_span in char_spans[1:]:
            if next_span["start"] <= current_span["end"]:
                # Merge overlapping spans
                current_span = {
                    "start": min(current_span["start"], next_span["start"]),
                    "end": max(current_span["end"], next_span["end"]),
                    "text": original[min(current_span["start"], next_span["start"]):
                                   max(current_span["end"], next_span["end"])]
                }
            else:
                merged_char_spans.append(current_span)
                current_span = next_span
        merged_char_spans.append(current_span)
    
    # Add tags
    result = original
    offset = 0
    for span in merged_char_spans:
        start = span["start"] + offset
        end = span["end"] + offset
        hallucination_text = result[start:end]
        marked_text = f"<hallucination>{hallucination_text}</hallucination>"
        result = result[:start] + marked_text + result[end:]
        offset += len(marked_text) - len(hallucination_text)
    
    # Construct image path
    image_path = ""
    if sample.get('image'):
        image_name = sample['image']
        image_path = image_name
    
    return {
        "id": f"mhal_{index}",
        "image_path": image_path,
        "original_solution": None,
        "test_solution": original,
        "hallucinated_solution": result,
        "hallucination_spans": merged_char_spans,
        "prompt": prompt
    }

def process_dataset(num_samples: int = 500):
    """Process mhal-detect dataset
    
    Args:
        num_samples (int): Number of samples to process, default is 500.
                           90% are hallucinated samples, 10% are clean samples.
    """
    print("Starting to load mhal-detect dataset...")
    
    # Calculate required number of hallucinated and clean samples
    required_hallucination_samples = int(num_samples * 0.9)
    required_clean_samples = num_samples - required_hallucination_samples
    
    print(f"Target total samples: {num_samples}")
    print(f"Target hallucinated samples: {required_hallucination_samples}")
    print(f"Target clean samples: {required_clean_samples}")
    
    # Load dataset using absolute path
    dataset_path = os.path.join(build_dir, 'data', 'mhal-detect', 'val_raw.json')
    print(f"Attempting to load dataset from: {dataset_path}")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        return
    
    print(f"The original dataset has {len(dataset)} entries")
    
    # Filter samples that contain images
    dataset_with_images = [sample for sample in dataset if sample.get('image')]
    print(f"Among them, {len(dataset_with_images)} entries contain images")
    
    if len(dataset_with_images) < num_samples:
        print(f"Warning: The number of entries containing images ({len(dataset_with_images)}) is less than the requested sample size ({num_samples})")
        dataset = dataset_with_images
    else:
        # Randomly select the specified number of samples containing images
        random.seed(42)
        dataset = random.sample(dataset_with_images, num_samples)
    
    # Process data
    processed_data = []
    for index, sample in enumerate(tqdm(dataset, desc="Processing data")):
        processed_sample = process_mhal_sample(sample, index)
        # For clean samples, set original_solution
        if not processed_sample['hallucination_spans']:
            processed_sample['original_solution'] = processed_sample['test_solution']
        processed_data.append(processed_sample)
    
    # Separate samples with and without hallucinations
    samples_with_hallucination = [s for s in processed_data if s['hallucination_spans']]
    samples_without_hallucination = [s for s in processed_data if not s['hallucination_spans']]
    
    print(f"\nPreliminary processing completed, found:")
    print(f"Hallucinated samples: {len(samples_with_hallucination)} entries")
    print(f"Clean samples: {len(samples_without_hallucination)} entries")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Select samples based on ratio
    final_hallucination_samples = []
    if len(samples_with_hallucination) > required_hallucination_samples:
        final_hallucination_samples = random.sample(samples_with_hallucination, required_hallucination_samples)
    else:
        final_hallucination_samples = samples_with_hallucination + random.choices(
            samples_with_hallucination,
            k=required_hallucination_samples - len(samples_with_hallucination)
        )
    
    final_clean_samples = []
    if len(samples_without_hallucination) > required_clean_samples:
        final_clean_samples = random.sample(samples_without_hallucination, required_clean_samples)
    else:
        final_clean_samples = samples_without_hallucination + random.choices(
            samples_without_hallucination,
            k=required_clean_samples - len(samples_without_hallucination)
        )
    
    # Merge final data
    final_processed_data = final_hallucination_samples + final_clean_samples
    random.shuffle(final_processed_data)  # Shuffle order randomly
    
    print(f"\nFinal data statistics:")
    print(f"Hallucinated samples: {len(final_hallucination_samples)} entries")
    print(f"Clean samples: {len(final_clean_samples)} entries")
    print(f"Total samples: {len(final_processed_data)} entries")
    
    # Set local image directory and output directory
    local_image_dir = os.path.join(build_dir, 'images', 'mhal')
    image_output_dir = os.path.join(build_dir, '..', 'evaluate', 'images', 'mhal')
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Clear image output directory
    for filename in os.listdir(image_output_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(image_output_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")
    
    # Start copying images from local directory to output directory
    print("\nStarting to copy images from local directory...")
    successful_copies = 0
    failed_copies = []
    
    for sample in tqdm(final_processed_data, desc="Copying images"):
        if not sample.get('image_path'):
            continue
            
        img_filename = os.path.basename(sample['image_path'])
        src_path = os.path.join(local_image_dir, img_filename)
        dst_path = os.path.join(image_output_dir, img_filename)
        
        if not os.path.exists(src_path):
            print(f"Warning: Local image does not exist: {src_path}")
            failed_copies.append(img_filename)
            continue
            
        try:
            import shutil
            shutil.copy2(src_path, dst_path)
            successful_copies += 1
        except Exception as e:
            print(f"Error copying image {img_filename}: {str(e)}")
            failed_copies.append(img_filename)
    
    print(f"\nImage copy completed:")
    print(f"Successfully copied: {successful_copies} images")
    print(f"Copy failed: {len(failed_copies)} images")
    if failed_copies:
        print("\nThe following images failed to copy:")
        for img in failed_copies[:10]:
            print(f"- {img}")
        if len(failed_copies) > 10:
            print(f"... and {len(failed_copies) - 10} more images not displayed")
    
    # Modify output path
    output_path = os.path.join(build_dir, '..', 'evaluate', 'data', 'processed_mhal_dataset_new.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing completed!")
    print(f"Data has been saved to: {output_path}")
    print(f"Images have been saved to: {image_output_dir}")

def test_hallucination_tagging():
    """Test the logic for adding hallucination tags"""
    test_cases = [
        {
            "name": "Basic Test - Using original and corrected text",
            "original": "The couch is situated in the center of the room.",
            "corrected": "The couch is in the room.",
            "expected_spans": [
                {"start": 3, "end": 4, "text": "situated"},
                {"start": 5, "end": 8, "text": "in the center"}
            ]
        },
        {
            "name": "Continuous Hallucination Test - Using original and corrected text",
            "original": "The image shows a living room with a large window. The walls are painted blue.",
            "corrected": "The image contains a window. The walls are white.",
            "expected_spans": [
                {"start": 2, "end": 7, "text": "shows a living room with"},
                {"start": 8, "end": 9, "text": "large"},
                {"start": 13, "end": 15, "text": "painted blue"}
            ]
        },
        {
            "name": "Directly Using Spans Test",
            "text": "The kitchen features modern appliances and granite countertops. The cabinets are wooden.",
            "spans": [
                {"start": 2, "end": 5, "text": "kitchen features modern"},
                {"start": 8, "end": 10, "text": "granite countertops"}
            ]
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest case: {test_case['name']}")
        
        if "original" in test_case and "corrected" in test_case:
            # Use extract_hallucination_spans_mhal to extract spans
            print("Original text:", test_case["original"])
            print("Corrected text:", test_case["corrected"])
            spans = extract_hallucination_spans_mhal(test_case["original"], test_case["corrected"])
            text = test_case["original"]
        else:
            # Directly use provided spans
            print("Text:", test_case["text"])
            spans = test_case["spans"]
            text = test_case["text"]
            
        print("Hallucination spans:", spans)
        
        # Use tagging method
        tagged = add_hallucination_tags_mhal(text, spans)
        print("\nTagged result:", tagged)
        
        # Verify results
        words = split_into_words(text)
        for span in spans:
            hallucinated_text = " ".join(words[span["start"]:span["end"]])
            if hallucinated_text.strip() and f"<hallucination>{hallucinated_text}</hallucination>" not in tagged:
                print(f"Warning: Expected tagged text not found: {hallucinated_text}")
        
        if "expected_spans" in test_case:
            # Verify extracted spans match expectations
            for expected_span in test_case["expected_spans"]:
                found = False
                for span in spans:
                    if span["start"] == expected_span["start"] and span["end"] == expected_span["end"]:
                        found = True
                        break
                if not found:
                    print(f"Warning: Expected span not found: {expected_span}")

def test_specific_case():
    """Test the processing logic for a specific sample"""
    # Test sample
    test_sample = {
        "question": "<image>\nCompose a comprehensive description of the image...",
        "response": "The image features a wooden bench placed on the side of a boat, which is docked near a body of water. The bench is adorned with stripes, giving it a unique appearance. There are two people sitting on the bench, enjoying the scenic view of the water and possibly engaging in conversation or relaxation. The overall atmosphere of the image is peaceful and serene, with the sound of the water and the presence of the boat adding to the tranquil setting.",
        "image": "COCO_val2014_000000292123.jpg",
        "annotations": [
            {
                "start": 0,
                "end": 101,
                "text": "The image features a wooden bench placed on the side of a boat, which is docked near a body of water.",
                "label": "ACCURATE"
            },
            {
                "start": 102,
                "end": 167,
                "text": "The bench is adorned with stripes, giving it a unique appearance.",
                "label": "INACCURATE"
            },
            {
                "start": 168,
                "end": 301,
                "text": "There are two people sitting on the bench, enjoying the scenic view of the water and possibly engaging in conversation or relaxation.",
                "label": "INACCURATE"
            },
            {
                "start": 302,
                "end": 449,
                "text": "The overall atmosphere of the image is peaceful and serene, with the sound of the water and the presence of the boat adding to the tranquil setting",
                "label": "ANALYSIS"
            }
        ]
    }
    
    # Process sample
    processed = process_mhal_sample(test_sample, 0)
    
    # Print processing results
    print("\nOriginal text:")
    print(test_sample["response"])
    print("\nAnnotation information:")
    for ann in test_sample["annotations"]:
        print(f"Label: {ann['label']}")
        print(f"Text: {ann['text']}")
        print(f"Range: {ann['start']}-{ann['end']}")
    
    print("\nProcessed text:")
    print(processed["hallucinated_solution"])
    print("\nHallucination spans:")
    for span in processed["hallucination_spans"]:
        print(f"Start: {span['start']}, End: {span['end']}, Text: {span['text']}")
    
    # Verify processing results
    # 1. Check if only INACCURATE labels were processed
    inaccurate_texts = [ann["text"] for ann in test_sample["annotations"] if ann["label"] == "INACCURATE"]
    print("\nVerification results:")
    print("1. INACCURATE label texts:")
    for text in inaccurate_texts:
        print(f"- {text}")
        # Check if all were correctly tagged
        if f"<hallucination>{text}</hallucination>" not in processed["hallucinated_solution"]:
            print(f"Warning: Expected tagged text not found: {text}")
    
    # 2. Check if other label texts were incorrectly tagged
    other_texts = [ann["text"] for ann in test_sample["annotations"] if ann["label"] != "INACCURATE"]
    print("\n2. Non-INACCURATE label texts:")
    for text in other_texts:
        print(f"- {text}")
        if f"<hallucination>{text}</hallucination>" in processed["hallucinated_solution"]:
            print(f"Warning: Found text that should not be tagged: {text}")
    
    # 3. Check continuity of spans
    print("\n3. Check continuity of spans:")
    for i in range(len(processed["hallucination_spans"]) - 1):
        current_span = processed["hallucination_spans"][i]
        next_span = processed["hallucination_spans"][i + 1]
        if current_span["end"] > next_span["start"]:
            print(f"Warning: Found overlapping spans: {current_span} and {next_span}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process mhal-detect dataset')
    parser.add_argument('--num_samples', type=int, default=500,
                      help='Number of samples to process (default: 500)')
    parser.add_argument('--test', action='store_true',
                      help='Run tagging test')
    parser.add_argument('--test_specific', action='store_true',
                      help='Run specific sample test')
    
    args = parser.parse_args()
    
    if args.test:
        test_hallucination_tagging()
    elif args.test_specific:
        test_specific_case()
    else:
        process_dataset(args.num_samples)