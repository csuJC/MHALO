import os
import json
from tqdm import tqdm
from openai import OpenAI
from typing import Optional, Dict, List, Union
import sys
import argparse
from dotenv import load_dotenv
from config import (
    DATASET_CONFIGS, 
    DatasetConfig, 
    Criteria_SYSTEM_MESSAGE,
    BASIC_COMPRESSED_SYSTEM_MESSAGE_2_SHOT,
    BASIC_MATH_COMPRESSED_SYSTEM_MESSAGE_2_SHOT,
    MATH_Criteria_SYSTEM_MESSAGE,
    Analyze_then_judge_SYSTEM_MESSAGE,
    MATH_Analyze_then_judge_SYSTEM_MESSAGE
)
from utils import (
    encode_image,
    prepare_evaluation_sample,
    calculate_f1m_score,
    calculate_step_f1m_score,
    print_evaluation_result,
    print_evaluation_summary,
    call_api_with_retry,
    calculate_iou_precision,
    extract_hallucination_spans,
)
import random
import re
import time
import pandas as pd
import concurrent.futures
from threading import Lock
import queue
import requests
import datetime
import shutil
from dataclasses import dataclass
# from local_models import LocalModel  # Import local model module

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# Get API configuration from environment variables
api_key = os.getenv('OPENAI_API_KEY')
api_base = os.getenv('OPENAI_API_BASE')
if not api_key or not api_base:
    raise ValueError("Please set environment variable OPENAI_API_KEY and OPENAI_API_BASE")

client = OpenAI(api_key=api_key, base_url=api_base)

# Add global locks for synchronized writing
results_lock = Lock()
progress_lock = Lock()
metrics_lock = Lock()

@dataclass
class DatasetConfig:
    name: str  # Dataset name
    data_path: str  # Data file path
    output_path: str  # Output file path
    system_message: str  # System prompt
    user_prompt_template: str  # User prompt template
    tmp_dir: str = ""  # Temporary directory path

def save_results_to_csv(all_results: Dict[str, List[Dict]], model: str, prompt_method: str = "vanilla", output_path: str = "results/tmp/evaluation_summary.csv"):
    """Save evaluation results of all datasets to a CSV file"""
    
    # Calculate metrics for each dataset
    dataset_metrics = {}
    total_samples = 0
    total_success = 0
    weighted_f1m_sum = 0
    weighted_iou_h_sum = 0
    
    # Collect metrics for each dataset
    for dataset_name, results in all_results.items():
        if not results:
            continue
            
        dataset_total = 0
        dataset_success = 0
        f1m_values = []
        iou_h_values = []
        
        for item in results:
            if isinstance(item, dict) and 'status' in item:
                if item['status'] == 'success':
                    dataset_success += 1
                    # Collect F1M and IOU metrics
                    f1m_values.append(item.get('metrics', {}).get('f1m', 0.0))
                    iou_h_values.append(item.get('metrics', {}).get('iou_h', 0.0))
        
        dataset_total = len(results)
        total_samples += dataset_total
        total_success += dataset_success
        
        # Calculate weighted metrics
        if dataset_total > 0:
            weighted_f1m_sum += (dataset_success / dataset_total) * sum(f1m_values) / len(f1m_values) if f1m_values else 0
            weighted_iou_h_sum += (dataset_success / dataset_total) * sum(iou_h_values) / len(iou_h_values) if iou_h_values else 0
        
        dataset_metrics[dataset_name] = {
            'total': dataset_total,
            'success': dataset_success,
            'f1m': sum(f1m_values) / len(f1m_values) if f1m_values else 0,
            'iou_h': sum(iou_h_values) / len(iou_h_values) if iou_h_values else 0
        }
    
    # Save metrics to CSV
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Total Samples', 'Successful Samples', 'F1M', 'IOU_H'])
        for dataset_name, metrics in dataset_metrics.items():
            writer.writerow([dataset_name, metrics['total'], metrics['success'], metrics['f1m'], metrics['iou_h']])
    
    print(f"Results saved to {output_path}")

def prepare_dataset(dataset, dataset_name: str, sample_limit: Optional[int] = None):
    """Prepare evaluation dataset
    
    Args:
        dataset: Original dataset
        dataset_name: Dataset name
        sample_limit: Sample count limit
    
    Returns:
        selected_indices: Set of selected sample indices
        has_hallucination_indices: Set of indices for samples with hallucination
    """
    if sample_limit is not None and sample_limit % 10 != 0:
        if sample_limit == 2:
            pass
        else:
            raise ValueError("sample_limit must be a multiple of 10")
    
    if dataset_name == "test":  
        samples_with_hallucination = []
        samples_without_hallucination = []
        
        for idx, sample in enumerate(dataset):
            if sample.get('original_solution') is None:
                samples_with_hallucination.append(idx)
            else:
                samples_without_hallucination.append(idx)
        
        all_indices = samples_with_hallucination + samples_without_hallucination
        if sample_limit and sample_limit < len(all_indices):
            selected_indices = set(random.sample(all_indices, sample_limit))
        else:
            selected_indices = set(all_indices)
    
        has_hallucination_indices = set(samples_with_hallucination)
        
    else: 
        total_samples = sample_limit if sample_limit else len(dataset)
        
        if dataset_name == "mhal": 
            hallucination_count = int(total_samples * 0.9)
            no_hallucination_count = total_samples - hallucination_count
            
            samples_with_hallucination = []
            samples_without_hallucination = []
            
            for idx, sample in enumerate(dataset):
                if sample.get('original_solution') is None:
                    samples_with_hallucination.append(idx)
                else:
                    samples_without_hallucination.append(idx)
            

            selected_with_hallucination = set(random.sample(samples_with_hallucination, hallucination_count))
            selected_without_hallucination = set(random.sample(samples_without_hallucination, no_hallucination_count))
            
            selected_indices = selected_with_hallucination | selected_without_hallucination
            has_hallucination_indices = selected_with_hallucination
            
        else:  
            hallucination_count = int(total_samples * 0.9)
            no_hallucination_count = total_samples - hallucination_count
            
            all_indices = list(range(len(dataset)))

            no_hallucination_indices = set(random.sample(all_indices, no_hallucination_count))

            remaining_indices = list(set(all_indices) - no_hallucination_indices)
            has_hallucination_indices = set(random.sample(remaining_indices, hallucination_count))
            
            selected_indices = has_hallucination_indices | no_hallucination_indices
    
    return selected_indices, has_hallucination_indices

def evaluate_single_sample(args):

    sample, idx, dataset_config, has_hallucination_indices, client, progress_bar, model, max_annotation_retries, api_retry_limit, prompt_method = args
    
    try:

        result = prepare_evaluation_sample(sample, dataset_config.name)
        if result is None:
            with progress_lock:
                progress_bar.update(1)
            return None
        

        result["has_hallucination"] = idx in has_hallucination_indices
        

        if not result["has_hallucination"]:
            if result.get('original_solution'):
                result['gt_solution'] = result['original_solution']
                result['test_solution'] = result['original_solution']
            else:
                with progress_lock:
                    progress_bar.update(1)
                return None
        

        annotation_retry_count = 0
        while annotation_retry_count < max_annotation_retries:

            messages = [
                {
                    "role": "system",
                    "content": dataset_config.system_message
                },
                {
                    "role": "user",
                    "content": []
                }
            ]
            

            if prompt_method == "vanilla":
                messages[1]["content"].append({
                    "type": "text",
                    "text": dataset_config.user_prompt_template.format(
                        prompt=result['prompt'],
                        test_description=result['test_solution']
                    )
                })
            elif prompt_method == "Criteria": 
                original_system_message = dataset_config.system_message
                if dataset_config.name in ["geo_170k"]:
                    dataset_config.system_message = MATH_Criteria_SYSTEM_MESSAGE
                else:
                    dataset_config.system_message = Criteria_SYSTEM_MESSAGE
                
                messages[1]["content"].append({
                    "type": "text",
                    "text": dataset_config.user_prompt_template.format(
                        prompt=result['prompt'],
                        test_description=result['test_solution']
                    )
                })
                dataset_config.system_message = original_system_message 
            elif prompt_method == "2-shot":
                original_system_message = dataset_config.system_message
                if dataset_config.name in ["geo_170k"]:
                    dataset_config.system_message = BASIC_MATH_COMPRESSED_SYSTEM_MESSAGE_2_SHOT
                else:
                    dataset_config.system_message = BASIC_COMPRESSED_SYSTEM_MESSAGE_2_SHOT
                
                messages[1]["content"].append({
                    "type": "text",
                    "text": dataset_config.user_prompt_template.format(
                        prompt=result['prompt'],
                        test_description=result['test_solution']
                    )
                })
                dataset_config.system_message = original_system_message
            elif prompt_method == "Analyze-then-judge":  
                original_system_message = dataset_config.system_message
                if dataset_config.name == "geo_170k":
                    dataset_config.system_message = MATH_Analyze_then_judge_SYSTEM_MESSAGE
                else:
                    dataset_config.system_message = Analyze_then_judge_SYSTEM_MESSAGE
                
                messages[1]["content"].append({
                    "type": "text",
                    "text": dataset_config.user_prompt_template.format(
                        prompt=result['prompt'],
                        test_description=result['test_solution']
                    )
                })
                dataset_config.system_message = original_system_message  
            else:
                raise ValueError(f"Unknown prompt method: {prompt_method}")
            
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(result['image_path'])}"
                }
            })
            
            response = call_api_with_retry(client, model, messages, max_retries=api_retry_limit)
            if not response or not hasattr(response, 'choices') or not response.choices:
                print(f"Warning: API response format is incorrect")
                print(f"Response content: {response}")
                raise ValueError("API response format is incorrect")
                
            if not hasattr(response.choices[0], 'message') or not response.choices[0].message:
                print(f"Warning: API response is missing the message field")
                print(f"Response content: {response}")
                raise ValueError("API response is missing the message field")
                
            gpt4v_response = response.choices[0].message.content
            if not gpt4v_response:
                print(f"Warning: API response content is empty")
                raise ValueError("API response content is empty")
            
            print(f"\nModel original response:")
            print(gpt4v_response)
            
            result["api_original_response"] = gpt4v_response
            result["gpt4v_response"] = gpt4v_response
            tagged_response = gpt4v_response
            tmp_response = gpt4v_response
            


            # Extract content from <Tagged_Text> tags
            tagged_text_pattern = r'<Tagged_Text>(.*?)</Tagged_Text>'
            tagged_text_match = re.search(tagged_text_pattern, tmp_response, re.DOTALL)
            if tagged_text_match:
                tmp_response = tagged_text_match.group(1).strip()
                tagged_response = tmp_response
                gpt4v_response = tmp_response
                result["gpt4v_response"] = gpt4v_response
                print(f"\nExtracted <Tagged_Text> content:")
                print(gpt4v_response)
                


            pattern = r'<hallucination>(.*?)</hallucination>'
            
            # Remove all tags and compare word counts
            text_without_tags = re.sub(pattern, r'\1', tmp_response)
            text_without_tags = text_without_tags.strip()
            original_text = result['test_solution'].strip()
            
            # Compare word counts
            compare_result = (len(text_without_tags.split()) == len(original_text.split()))
            
            if not compare_result:
                print(f"Warning: Word count mismatch")
                print(f"Original text word count: {len(original_text.split())}")
                print(f"Processed text word count: {len(text_without_tags.split())}")
                print(f"Original text: {original_text}")
                print(f"Processed text: {text_without_tags}")

            if compare_result:
                break
            
            annotation_retry_count += 1
            if annotation_retry_count < max_annotation_retries:
                print(f"Warning: Sample {result['sample_id']} is not correctly annotated on the {annotation_retry_count}th attempt, retrying...")
                continue
        
        if not compare_result:
            print(f"Warning: Sample {result['sample_id']} is not correctly annotated after {max_annotation_retries} attempts")
            result["status"] = "not_correct_annotated"
            with progress_lock:
                progress_bar.update(1)
            return result
        
        # Calculate F1M scores and F1M_iou for different k values
        metrics = calculate_f1m_score(
            result["gt_solution"], 
            result["gpt4v_response"],
            pred_spans=result.get("pred_spans")
        )
        
        result["metrics"] = metrics
        result["details"] = metrics["details"]
        result["status"] = "success"
        
        # Print evaluation results
        print_evaluation_result(result, dataset_config.name)
        
    except Exception as e:
        print(f"\nSample {result['sample_id']} evaluation failed: {str(e)}")
        result["status"] = "error"
        result["error"] = str(e)
    
    with progress_lock:
        progress_bar.update(1)
    
    return result

def evaluate_dataset(
    dataset_config: DatasetConfig,
    sample_limit: Optional[int] = None,
    restart: bool = True,
    model: str = "gpt-4o-2024-11-20",
    max_workers: int = 16,
    max_annotation_retries: int = 3,
    api_retry_limit: int = 3,
    prompt_method: str = "vanilla",
    return_results: bool = False,
    result_dir: Optional[str] = None  
) -> Optional[List[Dict]]:
    """Evaluate a single dataset"""
    
    # If result_dir is not provided, create a new timestamp directory
    if result_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        safe_model_name = model.replace('/', '_')
        result_dir = f"results/{timestamp}_{safe_model_name}_{prompt_method}_{sample_limit}"
    
    # Create a dataset subdirectory in the shared directory
    dataset_dir = os.path.join(result_dir, dataset_config.name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Modify the detailed results save path
    detailed_output_path = os.path.join(dataset_dir, f"{dataset_config.name}_detailed_results.json")
    
    # Load dataset
    with open(dataset_config.data_path, 'r') as f:
        dataset = json.load(f)
    
    # Limit sample count
    if sample_limit is not None:
        dataset = dataset[:sample_limit]
    
    # Initialize or load result file
    if restart or not os.path.exists(detailed_output_path):
        evaluation_results = []
    else:
        with open(detailed_output_path, 'r') as f:
            evaluation_results = json.load(f)
    
    # Create progress bar
    total_samples = len(dataset)
    progress_bar = tqdm(total=total_samples, desc=f"Evaluating {dataset_config.name} dataset")
    
    # Update progress bar to current completed count
    progress_bar.update(len(evaluation_results))
    
    # Prepare dataset
    selected_indices, has_hallucination_indices = prepare_dataset(dataset, dataset_config.name, sample_limit)
    
    # Process remaining samples using a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, sample in enumerate(dataset[len(evaluation_results):]):
            args = (sample, idx, dataset_config, has_hallucination_indices, 
                   client, progress_bar, model, max_annotation_retries, 
                   api_retry_limit, prompt_method)
            future = executor.submit(evaluate_single_sample, args)
            futures.append(future)
        
        # Collect results
        success_count = len([r for r in evaluation_results if r.get('status') == 'success'])
        total_f1m = 0.0
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    evaluation_results.append(result)
                    if result.get('status') == 'success':
                        success_count += 1
                        total_f1m += result.get('metrics', {}).get('f1m', 0.0)
                    
                    # Save intermediate results to detailed results file
                    with results_lock:
                        with open(detailed_output_path, 'w') as f:
                            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
                    
                    # Update progress bar
                    with progress_lock:
                        progress_bar.update(1)
            except Exception as e:
                print(f"Error processing sample: {str(e)}")
    
    progress_bar.close()
    
    # Print evaluation summary
    print_evaluation_summary(evaluation_results, total_f1m, success_count, dataset_config)
    
    if return_results:
        return evaluation_results
    return None

def evaluate_all_datasets(
    sample_limit: Optional[int] = None,
    restart: bool = True,
    model: str = "gpt-4o-2024-11-20",
    prompt_method: str = "vanilla",
    max_workers: int = 16,
    max_annotation_retries: int = 3,
    api_retry_limit: int = 3
) -> None:
    """Evaluate all datasets"""
    all_results = {}
    
    # Create a timestamp directory in the shared directory
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    safe_model_name = model.replace('/', '_')
    result_dir = f"results/{timestamp}_{safe_model_name}_{prompt_method}_{sample_limit}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Iterate through all dataset configurations
    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        print(f"\nStarting evaluation of {dataset_name.upper()} dataset...")
        
        # Evaluate a single dataset and get results, passing in the shared result_dir
        results = evaluate_dataset(
            dataset_config=dataset_config,
            sample_limit=sample_limit,
            restart=restart,
            model=model,
            max_workers=max_workers,
            max_annotation_retries=max_annotation_retries,
            api_retry_limit=api_retry_limit,
            prompt_method=prompt_method,
            return_results=True,
            result_dir=result_dir  
        )
        
        # Save results to dictionary
        all_results[dataset_name] = results
    
    # Save summary results
    summary_path = os.path.join(result_dir, "evaluation_summary.csv")
    save_results_to_csv(all_results, model, prompt_method, summary_path)

def main():
    parser = argparse.ArgumentParser(description='Evaluate the performance of MLLM models')
    parser.add_argument('--dataset', type=str, default='all',
                      choices=['all'] + list(DATASET_CONFIGS.keys()),
                      help='The dataset to evaluate, default is "all" to evaluate all datasets')
    parser.add_argument('--sample_limit', type=int, default=50,
                      help='Limit on the number of samples for evaluation (must be a multiple of 10)')
    parser.add_argument('--restart', type=bool, default=True,
                      help='Whether to restart evaluation. If False, read data from existing result files')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20',
                      choices=['gpt-4o-2024-11-20',
                               'claude-3-5-sonnet-20241022', 
                               'gemini-1.5-pro-002', 
                               'qwen-vl-max-0809',
                               'abab7-chat-preview',
                               'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
                               'glm-4v',
                               'local/MiniCPM-V-2_6',
                               'local/InternVL2-Llama3-76B'],
                      help='Select the model to use, default is gpt-4o-2024-11-20')
    parser.add_argument('--prompt_method', type=str, default='vanilla',
                      choices=['vanilla', 'Criteria', '2-shot', 'Analyze-then-judge'],
                      help='Select the prompt method to use, default is vanilla')
    parser.add_argument('--max_workers', type=int, default=20,
                      help='Maximum number of worker threads, default is 16')
    parser.add_argument('--max_annotation_retries', type=int, default=3,
                      help='Maximum number of annotation retries, default is 3')
    parser.add_argument('--api_retry_limit', type=int, default=3,
                      help='Maximum number of API call retries, default is 3')
    args = parser.parse_args()
    
    # Check environment variables
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("Please set environment variable OPENAI_API_KEY")
    if not os.getenv('OPENAI_API_BASE'):
        raise ValueError("Please set environment variable OPENAI_API_BASE")
    
    # Validate sample_limit
    if args.sample_limit is not None and args.sample_limit % 10 != 0:
        if args.sample_limit == 2:
            pass
        else:       
            raise ValueError("sample_limit must be a multiple of 10")
    
    if args.dataset == 'all':
        # Evaluate all datasets
        evaluate_all_datasets(
            sample_limit=args.sample_limit,
            restart=args.restart,
            model=args.model,
            prompt_method=args.prompt_method,
            max_workers=args.max_workers,
            max_annotation_retries=args.max_annotation_retries,
            api_retry_limit=args.api_retry_limit
        )
    else:
        # Evaluate a single dataset
        dataset_config = DATASET_CONFIGS[args.dataset]
        
        # Create timestamp directory
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        safe_model_name = args.model.replace('/', '_')
        result_dir = f"results/{timestamp}_{safe_model_name}_{args.prompt_method}_{args.sample_limit}"
        
        print(f"\nStarting evaluation of {args.dataset.upper()} dataset...")
        evaluate_dataset(
            dataset_config=dataset_config,
            sample_limit=args.sample_limit,
            restart=args.restart,
            model=args.model,
            max_workers=args.max_workers,
            max_annotation_retries=args.max_annotation_retries,
            api_retry_limit=args.api_retry_limit,
            prompt_method=args.prompt_method,
            result_dir=result_dir
        )

if __name__ == "__main__":
    main()