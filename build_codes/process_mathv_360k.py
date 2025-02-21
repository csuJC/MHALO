import json
import os
import base64
import re
from pathlib import Path
import random
from openai import OpenAI
from tqdm import tqdm
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time
from typing import List, Dict, Any

# Define the prompt for generating CoT and injecting hallucinations
system_prompt_360k = """You are an expert at mathematical reasoning and visual hallucination injection. Your task has three parts:

Part 1 - Generate Original Solution:
1. Carefully analyze the image, question and answer
2. Create a detailed step-by-step solution with clear reasoning
3. Make sure the solution is accurate and matches the visual elements
4. Wrap this solution in <original_solution> tags

Part 2 - Analyze Hallucination Opportunities:
1. Analyze the original solution to identify what types of information are present:
   - Visual elements (objects, text, numbers)
   - Attributes (quantities, colors, shapes, positions)
   - Relations (comparisons, spatial relationships)
   - Reasoning steps (logic, calculations, formulas)
2. Based on the analysis, select appropriate types of hallucinations from the hallucination types list:

Content Hallucinations:
1. **Object**: Misidentify objects in the image
2. **OCR**: Misread text or numbers in the image

Attribute Hallucinations:
3. **Numerical Attribute**: Misread quantities, sizes, measurements
4. **Color Attribute**: Misidentify colors of objects
5. **Shape Attribute**: Misinterpret shapes of objects
6. **Spatial Attribute**: Misread positions, orientations, distances

Relation Hallucinations:
7. **Numerical Relations**: Misinterpret quantitative comparisons
8. **Size Relations**: Misread relative sizes between objects
9. **Spatial Relations**: Misinterpret positions between objects

Reasoning Hallucinations:
10. **Logical Errors**: Make mistakes in reasoning steps
11. **Calculation Errors**: Perform incorrect mathematical operations
12. **Knowledge Errors**: Apply incorrect formulas or concepts
13. **Query Misunderstanding**: Misunderstand the query intent and gives wrong or irrelevant answers

3. Write your analysis in <hallucination_analysis> tags, explaining what types of hallucinations would be natural to inject based on the content, when writing the hallucination types, please strictly choose from the above 12 types of hallucinations, use the identical format like "**Object**", "**OCR**", "**Numerical Attribute**", "**Color Attribute**", "**Shape Attribute**", "**Spatial Attribute**", "**Numerical Relations**", "**Size Relations**", "**Spatial Relations**", "**Logical Errors**", "**Calculation Errors**", "**Knowledge Errors**", "**Query Misunderstanding**".

Part 3 - Create Hallucinated Version:
1. Based on your analysis, create a version with plausible but incorrect visual details
2. Tag ALL hallucinated spans with <hallucination> tags, and the final answer should also be tagged when it is hallucinated
3. Wrap the hallucinated version in <hallucinated_solution> tags
4. Do not add explanatory text about the hallucinations,especially Please dont include anywords like"misidentified","misinterpreting".'misinterpreted"

Example:
Q: In the geometric diagram, what is the area of the triangle?
A: 12

<original_solution>
Let's solve this step by step:
1. Looking at the image, I see:
   - A right triangle drawn on a grid
   - Base length is 4 units
   - Height is 6 units
   - Right angle marked with a square symbol
2. To find the area of a triangle:
   Area = (base × height) ÷ 2
3. Plugging in our values:
   Area = (4 × 6) ÷ 2 = 24 ÷ 2 = 12
Therefore, the area is 12 square units.
</original_solution>

<hallucination_analysis>
The original solution contains:
1. Shape information (right triangle)
2. Numerical measurements (base and height)
3. Visual markers (square symbol)
4. Mathematical calculations
5. area formula (knowledge)

Suitable hallucination types:
**Shape Attribute**: modify the triangle type
**Numerical Attribute**: alter the measurements
**Knowledge Errors**: apply incorrect formulas
These would maintain solution plausibility while introducing controlled errors.
</hallucination_analysis>

<hallucinated_solution>
Let's solve this step by step:
1. Looking at the image, I see:
   - A <hallucination>isosceles triangle</hallucination> drawn on a grid
   - Base length is <hallucination>5 units</hallucination>
   - Height is <hallucination>4.8 units</hallucination>
   - Right angle marked with a square symbol
2. To find the area of a triangle:
   <hallucination>Area = (base × height)</hallucination>
3. Plugging in our values:
   <hallucination>Area = (5 × 4.8) = 24</hallucination>
Therefore, the area is <hallucination>24</hallucination> square units.
</hallucinated_solution>

Requirements:
1. ALWAYS provide all three parts: original solution, hallucination analysis, and hallucinated solution
2. ALWAYS tag ALL hallucinated spans with <hallucination> tags
3. Ensure tags are not nested, overlapping, or continuous. For example, when tagging "hello there," <hallucination>hello</hallucination><hallucination>there</hallucination> is not allowed. The correct format is <hallucination>hello there</hallucination>
4. Keep solutions detailed and specific
5. Do not explain or point out the hallucinations in the hallucinated solution
6. Start solutions with "Let's solve this step by step:" or "Let's analyze the image step by step:"

Remember: Success depends on proper tagging of EVERY hallucinated span and maintaining the solution structure!"""

# Convert the image to base64 encoding
def encode_image(image_path: str) -> str:
    """Convert image to base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Generate hallucinated answers with reasoning chains using GPT-4-Vision
def generate_solution(client: OpenAI, question: str, answer: str, image_path: str) -> tuple[str, str, str, str]:
    """Generate hallucinated answers with reasoning chains using GPT-4-Vision"""
    messages = [
        {
            "role": "system",
            "content": system_prompt_360k
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Question: {question}\nAnswer: {answer}"
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
        original, with_tag, without_tag = extract_solutions(api_response)
        return original, with_tag, without_tag, api_response
    except Exception as e:
        print(f"Error generating solution: {str(e)}")
        return None, None, None, None

# Extract original answers and tagged/untagged hallucinated answers from GPT response
def extract_solutions(response: str) -> tuple[str, str, str]:
    """Extract original answers and tagged/untagged hallucinated answers from GPT response"""
    # Extract original solution
    pattern_original = r'<original_solution>(.*?)</original_solution>'
    match_original = re.search(pattern_original, response, re.DOTALL)
    
    if not match_original:
        print("Original solution not found")
        print(response)
        return None, None, None
    
    original_solution = match_original.group(1).strip()
    
    # Extract tagged hallucinated solution
    pattern_with_tag = r'<hallucinated_solution>(.*?)</hallucinated_solution>'
    match_with_tag = re.search(pattern_with_tag, response, re.DOTALL)
    
    if not match_with_tag:
        print("Tagged hallucinated solution not found")
        print(response)
        return None, None, None
    
    solution_with_tag = match_with_tag.group(1).strip()
    
    if '<hallucination>' not in solution_with_tag:
        print("Hallucination tags not found")
        print(response)
        return None, None, None
    
    # Generate version without tags
    solution_without_tag = re.sub(r'<hallucination>(.*?)</hallucination>', r'\1', solution_with_tag)
    
    return original_solution, solution_with_tag, solution_without_tag

def generate_with_retry(client, question, answer, src_image):
    """Generate function with retry mechanism"""
    max_retries = 3
    retry_count = 0
    
    print(f"\nStarting data processing:")
    print(f"Question: {question[:100]}...")
    print(f"Answer: {answer[:100]}...")
    print(f"Image path: {src_image}")
    
    while retry_count < max_retries:
        try:
            original_solution, hallucinated_solution, test_solution, api_response = generate_solution(
                client, question, answer, str(src_image)
            )
            
            if original_solution and hallucinated_solution and test_solution:
                print("✓ Generation successful!")
                return original_solution, hallucinated_solution, test_solution, api_response, True
            
            print(f"✗ Generation attempt {retry_count + 1} failed, retrying...")
            retry_count += 1
            
            if retry_count == max_retries:
                print(f"! Maximum retry count ({max_retries}) reached, skipping this data\n")
                return None, None, None, None, False
                
        except Exception as e:
            print(f"\n✗ Generation attempt {retry_count + 1} failed:")
            print(f"Error message: {str(e)}")
            retry_count += 1
            if retry_count == max_retries:
                return None, None, None, None, False
    
    return None, None, None, None, False

# Function to process a single data item
def process_single_item(item: Dict[str, Any], 
                       client: OpenAI,
                       input_images_dir: Path,
                       output_images_dir: Path,
                       lock: threading.Lock) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a single data item
    
    Args:
        item: Data item to process
        client: OpenAI client
        input_images_dir: Input images directory
        output_images_dir: Output images directory
        lock: Thread lock
        
    Returns:
        tuple: (Processed data, Failed info)
    """
    try:
        if 'image' not in item:
            raise ValueError("Missing image field")
        
        src_image = input_images_dir / item['image']
        if not src_image.exists():
            raise FileNotFoundError(f"Image file does not exist: {src_image}")
        
        question = item['conversations'][0]['value'] if item.get('conversations') else ''
        answer = item['conversations'][1]['value'] if len(item.get('conversations', [])) > 1 else ''
        
        original_solution, hallucinated_solution, test_solution, api_response, success = generate_with_retry(
            client, question, answer, str(src_image)
        )
        
        if not success:
            raise Exception("Generation failed")
        
        # Use thread lock to protect file copying operation
        with lock:
            relative_image_path = item['image']
            dst_image = output_images_dir / relative_image_path
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_image, dst_image)
        
        processed_item = {
            'image': relative_image_path,
            'question': question,
            'original_solution': original_solution,
            'hallucinated_solution': hallucinated_solution,
            'test_solution': test_solution,
            'answer': answer,
            'api_response': api_response
        }
        
        return processed_item, None
        
    except Exception as e:
        failed_info = {
            "question": question if 'question' in locals() else "",
            "image_path": str(src_image) if 'src_image' in locals() else "",
            "error": str(e)
        }
        return None, failed_info

# Process the dataset and generate versions with hallucinations
def process_dataset(num_samples: int = 10, num_threads: int = 4):
    """Process the dataset and generate versions with hallucinations
    
    Args:
        num_samples (int): Number of samples to generate, default 10
        num_threads (int): Number of threads to use, default 4
    """
    # Set OpenAI client
    client = OpenAI(
        base_url="https://one-api.glm.ai/v1",
        api_key = 'sk-ECLer7uvOSXsLd9fC7110b72A15f47848e418b2049C69361'
    )
    
    # Set input/output paths
    input_file = Path('build/data/mathv/train_samples_all_tuning.json')
    input_images_dir = Path('build/images/mathv')
    output_images_dir = Path('evaluate/images/mathv_360k')
    output_file = Path('evaluate/data/processed_mathv_360k.json')
    
    # If output file exists, delete it
    if output_file.exists():
        output_file.unlink()
    
    # If output image directory exists, delete it
    if output_images_dir.exists():
        shutil.rmtree(output_images_dir)
    
    # Create output directory
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Read original dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get total sample count
    total_samples = len(data)
    print(f"Original dataset contains {total_samples} entries")
    print(f"Target to generate {num_samples} entries")
    print(f"Using {num_threads} threads in parallel processing")
    
    # Initialize data structure
    processed_data = []
    failed_samples = []
    lock = threading.Lock()
    
    # Create progress bar
    pbar = tqdm(total=num_samples, desc="Processing MathV dataset")
    
    # Current processing data index
    current_index = 0
    batch_size = min(num_threads * 2, num_samples)  # Initial batch size is 2 times the thread count
    
    while len(processed_data) < num_samples and current_index < total_samples:
        # Calculate the number of entries to process in this batch
        remaining_samples = num_samples - len(processed_data)
        current_batch_size = min(batch_size, remaining_samples)
        
        # Ensure there is enough data to process
        if current_index + current_batch_size > total_samples:
            current_batch_size = total_samples - current_index
            
        if current_batch_size <= 0:
            break
            
        current_batch = data[current_index:current_index + current_batch_size]
        
        print(f"\nCurrent processing progress:")
        print(f"Successfully processed: {len(processed_data)}/{num_samples}")
        print(f"Cumulative failed count: {len(failed_samples)}")
        print(f"Current batch size: {len(current_batch)}")
        print(f"Current processing index: {current_index}")
        
        # Use thread pool to process data
        successful_in_batch = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_item = {
                executor.submit(
                    process_single_item, 
                    item, 
                    client,
                    input_images_dir,
                    output_images_dir,
                    lock
                ): item for item in current_batch
            }
            
            for future in as_completed(future_to_item):
                processed_item, failed_info = future.result()
                
                with lock:
                    if processed_item:
                        if len(processed_data) < num_samples:
                            processed_data.append(processed_item)
                            successful_in_batch += 1
                            pbar.update(1)
                    elif failed_info:
                        failed_info["idx"] = current_index + len(failed_samples)
                        failed_samples.append(failed_info)
                        print(f"\nError processing sample: {failed_info['error']}")
        
        # Update current index to ensure forward movement
        current_index += current_batch_size
        
        # Dynamically adjust batch size
        if successful_in_batch == 0:
            # If this batch completely fails, reduce batch size to reduce resource waste
            batch_size = max(num_threads, batch_size // 2)
        elif successful_in_batch == current_batch_size:
            # If this batch completely succeeds, slightly increase batch size
            batch_size = min(batch_size * 2, num_samples - len(processed_data))
        
        # If reached end of dataset but target count not reached
        if len(processed_data) < num_samples and current_index >= total_samples:
            print(f"\nWarning: Reached end of dataset but target count not reached!")
            print(f"Target count: {num_samples}")
            print(f"Actually generated: {len(processed_data)}")
            break
    
    pbar.close()
    
    # Print processing statistics
    print(f"\nSuccessfully processed samples: {len(processed_data)}")
    print(f"Failed samples: {len(failed_samples)}")
    print(f"Total attempted data entries: {current_index}")
    
    # Save processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # Save failed samples if any
    if failed_samples:
        failed_samples_path = "evaluate/data/failed_mathv_360k_samples.json"
        with open(failed_samples_path, "w", encoding="utf-8") as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        print(f"Failed sample information saved to: {failed_samples_path}")
    
    if len(processed_data) < num_samples:
        print(f"\nWarning: Target count not reached!")
        print(f"Target count: {num_samples}")
        print(f"Actually generated: {len(processed_data)}")
        print(f"Possible reasons: Insufficient samples in dataset or high failure rate")
    else:
        print(f"Processing completed!")
        print(f"Data saved to: {output_file}")
        print(f"Images saved to: {output_images_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process math dataset and generate versions with hallucinations')
    parser.add_argument('--num_samples', type=int, default=500,
                      help='Number of samples to generate (default: 500)')
    parser.add_argument('--num_threads', type=int, default=8,
                      help='Number of threads to use (default: 8)')
    
    args = parser.parse_args()
    process_dataset(args.num_samples, args.num_threads) 

