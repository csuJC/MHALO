import json
import os
import base64
import re
from pathlib import Path
import random
from openai import OpenAI
from tqdm import tqdm
import shutil
import copy
from config import system_prompt_170k
import sys
# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define prompt
HALLUCINATION_PROMPT = system_prompt_170k

def encode_image(image_path: str) -> str:
    """Convert image to base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_solutions(response: str) -> tuple[str, str]:
    """Extract original and hallucinated solutions from the response"""
    pattern_with_tag = r'<hallucinated_solution>(.*?)</hallucinated_solution>'
    match_with_tag = re.search(pattern_with_tag, response, re.DOTALL)
    
    if not match_with_tag:
        return None, None
    
    solution_with_tag = match_with_tag.group(1).strip()
    
    # Check if it contains <hallucination> tag
    if '<hallucination>' not in solution_with_tag:
        return None, None
    
    solution_without_tag = re.sub(r'<hallucination>(.*?)</hallucination>', r'\1', solution_with_tag)
    
    return solution_with_tag, solution_without_tag

def generate_hallucinated_solution(client: OpenAI, question: str, original_solution: str, image_path: str) -> tuple[str, str, str]:
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE')
    if not api_key or not api_base:
        raise ValueError("Please set the environment variables OPENAI_API_KEY and OPENAI_API_BASE")

    client = OpenAI(api_key=api_key, base_url=api_base)

    messages = [
        {
            "role": "system",
            "content": HALLUCINATION_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Question: {question}\n\nOriginal solution:\n{original_solution}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        api_response = response.choices[0].message.content
        api_response_copy = copy.deepcopy(api_response)

        print("api_response", api_response)
        
        # Add safety check
        match = re.search(r'<hallucinated_solution>(.*?)</hallucinated_solution>', api_response, re.DOTALL)
        if not match:
            print("Warning: hallucinated_solution tag not found in API response")
            return None, None, api_response_copy
            
        hallucinated_solution, solution_without_tag = extract_solutions(api_response)
        return hallucinated_solution, solution_without_tag, api_response_copy
    except Exception as e:
        print(f"Error generating hallucinated solution: {str(e)}")
        return None, None, None

def generate_with_retry(client, question, original_solution, src_image):
    """Generate function with retry mechanism"""
    max_retries = 6  # Increased to 6 retries
    retry_count = 0
    
    print(f"\nStart processing data:")
    print(f"Question: {question[:100]}...")  # Only show the first 100 characters
    print(f"Original solution: {original_solution[:100]}...")
    print(f"Image path: {src_image}")
    
    while retry_count < max_retries:
        try:
            hallucinated_solution, test_solution, api_response = generate_hallucinated_solution(
                client, question, original_solution, str(src_image)
            )

            print("hallucinated_solution", hallucinated_solution)
            print("test_solution", test_solution)
            print("api_response_new", api_response)
            
            if hallucinated_solution and test_solution:
                print(f"✓ Generation successful!")
                return hallucinated_solution, test_solution, api_response, True
            
            print(f"✗ Generation failed for the {retry_count + 1} time, retrying...")
            retry_count += 1
            
            if retry_count == max_retries:
                print(f"! Reached maximum retry count ({max_retries}), skipping this data\n")
                return None, None, None, False
                
        except Exception as e:
            print(f"\n✗ Generation failed:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            
            retry_count += 1
            if retry_count == max_retries:
                print(f"! Reached maximum retry count ({max_retries}), skipping this data\n")
                return None, None, None, False
    
    return None, None, None, False

def process_dataset(sample_size: int = 2000, overwrite: bool = False, batch_size: int = 4):
    """
    Process the Geo170K dataset to generate hallucinated answers
    
    Args:
        sample_size (int): Number of entries to process, default is 2000
        overwrite (bool): Whether to overwrite existing files, default is False
        batch_size (int): Batch size for parallel processing, default is 4
    """
    # Set up OpenAI client
    clients = [OpenAI(
        base_url= "https://one-api.glm.ai/v1",
        api_key = 'sk-ECLer7uvOSXsLd9fC7110b72A15f47848e418b69361'
    ) for _ in range(batch_size)]
    
    # Set output paths
    input_file = Path('/workspace/yishuo_sft/cys/mllm_meta_evl/build/data/Geo170K/qa_tuning.json')
    input_images_dir = Path('/workspace/yishuo_sft/cys/mllm_meta_evl/build/images/170k')
    output_images_dir = Path('/workspace/yishuo_sft/cys/mllm_meta_evl/evaluate/images/geo_170k')
    output_file = Path('/workspace/yishuo_sft/cys/mllm_meta_evl/evaluate/data/processed_geo_170k.json')
    
    # If output image directory exists, delete it
    if output_images_dir.exists():
        shutil.rmtree(output_images_dir)
    
    # Create output directory
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Read the original dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Output dataset information
    total_samples = len(data)
    print(f"The original dataset has {total_samples} entries")
    print(f"Target to generate {sample_size} entries")
    print(f"Using {batch_size} parallel processes")

    # Select the first sample_size entries
    data = data[:sample_size]
    
    # Process each data entry
    processed_data = []
    failed_samples = []  # To record failed samples
    pbar = tqdm(total=sample_size, desc="Processing Geo170K dataset")

    def process_batch(batch_items, client_idx):
        batch_results = []
        for item in batch_items:
            try:
                # Check for image field
                if 'image' not in item:
                    raise ValueError("Missing image field")
                
                src_image = input_images_dir / item['image']
                if not src_image.exists():
                    raise FileNotFoundError(f"Image file does not exist: {src_image}")
                
                # Get question and original answer
                question = item['conversations'][0]['value'] if item.get('conversations') else ''
                original_solution = item['conversations'][1]['value'] if len(item.get('conversations', [])) > 1 else ''
                
                # Use GPT to generate hallucinated answer
                hallucinated_solution, test_solution, api_response, success = generate_with_retry(
                    clients[client_idx], question, original_solution, str(src_image)
                )
                
                if not success:
                    raise Exception("Generation failed")
                
                # Copy image to output directory
                relative_image_path = item['image']  # Keep the original relative path
                dst_image = output_images_dir / relative_image_path
                dst_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_image, dst_image)
                
                # Build new data structure
                batch_results.append({
                    'image': relative_image_path,
                    'question': question,
                    'original_solution': original_solution,
                    'hallucinated_solution': hallucinated_solution,
                    'test_solution': test_solution,
                    'api_response': api_response
                })
                pbar.update(1)
            except Exception as e:
                failed_samples.append({
                    "idx": len(processed_data) + len(failed_samples) + 1,
                    "question": question if 'question' in locals() else "",
                    "image_path": str(src_image) if 'src_image' in locals() else "",
                    "error": str(e)
                })
                print(f"\nError processing sample: {str(e)}")
        return batch_results

    # Split data into batches
    batch_data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for batch_idx, batch in enumerate(batch_data):
            client_idx = batch_idx % batch_size
            future = executor.submit(process_batch, batch, client_idx)
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            batch_results = future.result()
            processed_data.extend(batch_results)

    pbar.close()

    # Output processing statistics
    print(f"\nSuccessfully processed samples: {len(processed_data)}")
    print(f"Failed samples: {len(failed_samples)}")

    # Decide file name based on overwrite parameter
    if not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{timestamp}_{sample_size}{output_file.suffix}")

    # Save processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # If there are failed samples, save failure records
    if failed_samples:
        failed_samples_path = "evaluate/data/failed_geo_170k_samples.json"
        with open(failed_samples_path, "w", encoding="utf-8") as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        print(f"Failed sample information has been saved to: {failed_samples_path}")

    print(f"Processing completed!")
    print(f"Data has been saved to: {output_file}")
    print(f"Images have been saved to: {output_images_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Processing Geo170K dataset, generating hallucinated solutions')
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of entries to process (default: 2000)')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing files (default: no)')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for parallel processing (default: 4)')
    
    args = parser.parse_args()
    process_dataset(args.num_samples, args.overwrite, args.batch_size) 