import os
import base64
from typing import Dict, List, Optional, Union
import re
from openai import OpenAI
import time
from scipy.optimize import linear_sum_assignment
import numpy as np


def extract_hallucination_spans(text: str) -> List[Dict[str, Union[int, str]]]:
    """Extract hallucination spans from text
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of dictionaries containing:
            - start: Start position (word count)
            - end: End position (word count)
            - text: Hallucinated text content
    """
    if not isinstance(text, str):
        print(f"Warning: Input text is not string type: {type(text)}")
        return []

    pattern = r'<hallucination>(.*?)</hallucination>'
    words = text.split()
    spans = []
    
    for match in re.finditer(pattern, text):
        hallucination_text = match.group(1)
        start_char = match.start()
        prefix_text = text[:start_char]
        start_word = len(prefix_text.split())
        hallucination_words = hallucination_text.split()
        end_word = start_word + len(hallucination_words)
        
        spans.append({
            'start': start_word,
            'end': end_word,
            'text': hallucination_text
        })
    
    spans.sort(key=lambda x: x['start'])
    return spans


def call_api_with_retry(client, model_name, messages, max_retries=3, retry_delay=5, temperature=0):
    """API call with retry mechanism"""
    for attempt in range(max_retries):
        try:
            print(f"\nAttempting to call API ({attempt + 1} times)...")
            print(f"Model: {model_name}")
            print(f"Message length: {len(str(messages))} characters")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1000,
                top_p=0.9,
                temperature=temperature,
                presence_penalty=0.3,
                frequency_penalty=0.3
            )
            
            # Verify response format
            if not response:
                raise ValueError("API returned empty response")
                
            if not hasattr(response, 'choices') or not response.choices:
                raise ValueError("API response missing choices field")
                
            if not hasattr(response.choices[0], 'message'):
                raise ValueError("API response missing message field")
                
            if not response.choices[0].message.content:
                raise ValueError("API response content is empty")
                
            print(f"API call successful!")
            return response
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            print(f"\n {attempt + 1} times call failed:")
            print(f"Error type: {error_type}")
            print(f"Error message: {error_msg}")
            
            if attempt == max_retries - 1:  # The last attempt
                raise Exception(
                    f"API call failed (has retried {max_retries} times):\n"
                    f"- Error type: {error_type}\n"
                    f"- Error message: {error_msg}\n"
                    f"- Model: {model_name}\n"
                    f"- Request content length: {len(str(messages))} characters"
                )
            
            print(f"{retry_delay} seconds later, attempt {attempt + 2}...")
            time.sleep(retry_delay)
            continue


def encode_image(image_path: str) -> str:
    """encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_evaluation_sample(sample: Dict, dataset_type: str) -> Optional[Dict]:
    """prepare evaluation sample"""
    if dataset_type in ["geo_170k", "mathv_360k", "multimath_300k"]:  # math problem dataset processing logic
        # Check required fields for math problem dataset
        required_fields = ['image', 'hallucinated_solution', 'test_solution']
        if dataset_type == "multimath_300k":
            required_fields.append('title')  # multimath_300k use title
        else:
            required_fields.append('question')  # other datasets use question
            
        if not all(field in sample for field in required_fields):
            print(f"Sample missing required fields: {[f for f in required_fields if f not in sample]}")
            return None
        
        # Build image path
        image_path = os.path.join('images', dataset_type, sample['image'])
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        # Special processing for multimath_300k dataset
        if dataset_type == "multimath_300k":
            return {
                "sample_id": sample['image'],  # Use image path as id
                "image_path": image_path,
                "original_solution": sample['original_solution'] if isinstance(sample, dict) else None,
                "gt_solution": sample['hallucinated_solution'],
                "test_solution": sample['test_solution'],
                "prompt": sample['title']  # multimath_300k use title
            }
        else:
            return {
                "sample_id": sample['image'],  # Use image path as id
                "image_path": image_path,
                "original_solution": sample.get('original_solution'),
                "gt_solution": sample['hallucinated_solution'],
                "test_solution": sample['test_solution'],
                "prompt": sample['question']  # other datasets use question
            }
    elif dataset_type in ["test", "mhal"]:  # test and mhal datasets use the same processing logic
        # Check required fields
        required_fields = ['image_path', 'hallucinated_solution', 'test_solution', 'prompt']
        if not all(field in sample for field in required_fields):
            print(f"Sample missing required fields: {[f for f in required_fields if f not in sample]}")
            return None
        
        # Build image path
        image_path = os.path.join('images', dataset_type, sample['image_path'])
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        # For test dataset, use the combination of image path and prompt as ID
        if dataset_type == "test":
            sample_id = f"{sample['image_path']}_{sample['prompt']}"
        else:
            sample_id = sample.get('id', sample['image_path'])
        
        return {
            "sample_id": sample_id,
            "image_path": image_path,
            "original_solution": sample.get('original_solution'),
            "gt_solution": sample['hallucinated_solution'],
            "test_solution": sample['test_solution'],
            "prompt": sample['prompt']
        }
    else:
        # Other datasets processing logic
        required_fields = ['id', 'image_path', 'hallucinated_solution', 'prompt']
        if not all(field in sample for field in required_fields):
            print(f"Sample missing required fields: {[f for f in required_fields if f not in sample]}")
            return None
        
        # Build image path
        image_path = os.path.join('images', dataset_type, sample['image_path'])
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        return {
            "sample_id": sample['id'],
            "image_path": image_path,
            "original_solution": sample.get('original_solution'),
            "gt_solution": sample['hallucinated_solution'],
            "test_solution": sample['test_solution'],
            "prompt": sample['prompt']
        }

def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for comparison
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # 1. Remove punctuation
    text = re.sub(r'[^\w\s°]', '', text)
    # 2. Normalize whitespace characters
    text = ' '.join(text.split())
    return text

def calculate_partial_match_precision(pred_span: Dict, gt_spans: List[Dict]) -> float:
    """Calculate the partial match precision of a single prediction span, based on word index"""
    pred_indices = set(range(pred_span["start"], pred_span["end"]))
    
    if not pred_indices:
        return 0.0
    
    # Normalize predicted text
    pred_text = normalize_text_for_comparison(pred_span["text"])
    pred_words = set(pred_text.split())
    
    if not pred_words:
        return 0.0
    
    max_overlap = 0
    for gt_span in gt_spans:
        # Normalize ground truth text
        gt_text = normalize_text_for_comparison(gt_span["text"])
        gt_words = set(gt_text.split())
        
        if not gt_words:
            continue
            
        overlap = len(pred_words & gt_words)
        max_overlap = max(max_overlap, overlap)
    
    return max_overlap / len(pred_words)

def calculate_partial_match_recall(gt_span: Dict, pred_spans: List[Dict]) -> float:
    """Calculate the partial match recall of a single ground truth span, based on word index"""
    gt_indices = set(range(gt_span["start"], gt_span["end"]))
    
    if not gt_indices:
        return 0.0
    
    # Normalize ground truth text
    gt_text = normalize_text_for_comparison(gt_span["text"])
    gt_words = set(gt_text.split())
    
    if not gt_words:
        return 0.0
    
    max_overlap = 0
    for pred_span in pred_spans:
        # Normalize predicted text
        pred_text = normalize_text_for_comparison(pred_span["text"])
        pred_words = set(pred_text.split())
        
        if not pred_words:
            continue
            
        overlap = len(gt_words & pred_words)
        max_overlap = max(max_overlap, overlap)
    
    return max_overlap / len(gt_words)

def calculate_f1m_score(ground_truth: str, prediction: str, pred_spans: List[Dict] = None) -> dict:
    """Calculate F1M score
    
    Args:
        ground_truth: Ground truth text
        prediction: Prediction text
        pred_spans: Pre-extracted prediction spans (optional)
        
    Returns:
        dict: Dictionary containing precision, recall, f1m, etc.
    """
    # Extract hallucination spans from prediction text
    if pred_spans is None:
        pred_spans = extract_hallucination_spans(prediction)
    
    # Extract hallucination spans from ground truth text
    gt_spans = extract_hallucination_spans(ground_truth)
    
    # Calculate precision and recall
    total_precision = 0
    total_recall = 0
    
    # Calculate precision for each prediction span
    for pred_span in pred_spans:
        precision = calculate_partial_match_precision(pred_span, gt_spans)
        total_precision += precision
    
    # Calculate recall for each ground truth span
    for gt_span in gt_spans:
        recall = calculate_partial_match_recall(gt_span, pred_spans)
        total_recall += recall
    
    # Calculate final precision and recall
    precision = total_precision / len(pred_spans) if pred_spans else 0.0
    recall = total_recall / len(gt_spans) if gt_spans else 0.0
    
    # Calculate F1 score
    f1m = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate F1M_iou
    iou_precision_g = calculate_iou_precision(pred_spans, gt_spans)
    iou_h = calculate_iou_precision(gt_spans, pred_spans)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1m": f1m,
        "iou_precision_g": iou_precision_g,
        "iou_h": iou_h,
        "details": {
            "pred_spans": pred_spans,
            "gt_spans": gt_spans
        }
    }

def extract_step_hallucinations(text: str) -> List[Dict[str, Union[int, str]]]:
    """Extract step-level hallucinations from text with <hallucination confidence=X> tags
    
    Args:
        text: Text with <hallucination confidence=X> tags
        
    Returns:
        List of dictionaries containing:
            - step: Step number
            - text: Hallucinated step text
            - confidence: Confidence score
    """
    spans = []
    # 修改正则表式以只匹配Step N部分
    pattern = r'<hallucination confidence=(\d+)>(Step \d+)</hallucination>'
    
    for match in re.finditer(pattern, text):
        confidence = int(match.group(1))
        step_text = match.group(2)
        
        # Extract step number
        step_num = int(re.search(r'Step (\d+)', step_text).group(1))
        
        # Get complete step content
        full_step_pattern = f"{step_text}[^S]*?"  # Match content before next Step
        full_step_match = re.search(full_step_pattern, text)
        if full_step_match:
            full_step_text = full_step_match.group(0).strip()
        else:
            full_step_text = step_text
        
        spans.append({
            "step": step_num,
            "text": full_step_text,
            "confidence": confidence
        })
    
    return spans

def calculate_step_f1m_score(gt_solution: str, pred_solution: str, confidence_threshold: int = 0) -> Dict:
    """Calculate step-level F1M score
    
    Args:
        gt_solution: Ground truth solution (with <hallucination> tags)
        pred_solution: Predicted solution (with <hallucination confidence=X> tags)
        confidence_threshold: Confidence threshold, only consider spans with confidence greater than this value
        
    Returns:
        Dictionary containing F1M score and other metrics
    """
    # Extract hallucinated steps from ground truth solution
    gt_pattern = r'<hallucination>(Step \d+)</hallucination>'
    gt_steps = []
    for match in re.finditer(gt_pattern, gt_solution):
        step_text = match.group(1)
        # Get complete step content
        full_step_pattern = f"{step_text}[^S]*?"
        full_step_match = re.search(full_step_pattern, gt_solution)
        if full_step_match:
            gt_steps.append(full_step_match.group(0).strip())
        else:
            gt_steps.append(step_text)
    gt_steps = set(gt_steps)
    
    # Extract predicted hallucinated steps, and filter based on confidence threshold
    pred_steps = set()
    for step in extract_step_hallucinations(pred_solution):
        if step.get('confidence', 0) > confidence_threshold:
            pred_steps.add(step["text"])
    
    # Calculate precision and recall
    if not gt_steps and not pred_steps:
        return {
            "f1m": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "details": {
                "gt_steps": list(gt_steps),
                "pred_steps": list(pred_steps)
            }
        }
    
    if not pred_steps:
        return {
            "f1m": 0.0,
            "precision": 1.0,
            "recall": 0.0,
            "details": {
                "gt_steps": list(gt_steps),
                "pred_steps": list(pred_steps)
            }
        }
    
    if not gt_steps:
        return {
            "f1m": 0.0,
            "precision": 0.0,
            "recall": 1.0,
            "details": {
                "gt_steps": list(gt_steps),
                "pred_steps": list(pred_steps)
            }
        }
    
    # Calculate intersection
    correct_steps = gt_steps & pred_steps
    
    precision = len(correct_steps) / len(pred_steps)
    recall = len(correct_steps) / len(gt_steps)
    
    # Calculate F1M
    if precision + recall == 0:
        f1m = 0.0
    else:
        f1m = 2 * (precision * recall) / (precision + recall)
    
    return {
        "f1m": f1m,
        "precision": precision,
        "recall": recall,
        "details": {
            "gt_steps": list(gt_steps),
            "pred_steps": list(pred_steps)
        }
    }


def print_evaluation_result(result: Dict, dataset_type: str):
    """Print evaluation result for a single sample"""
    print(f"\nSample {result['sample_id']} evaluation result:")
    print(f"Ground Truth: {result['gt_solution']}")
    print(f"GPT-4V: {result['gpt4v_response']}")
    
        # Print basic metrics
    metrics = result['metrics']
    print(f"F1M: {metrics['f1m']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1M_iou (h): {metrics['iou_h']:.4f}")
    print(f"F1M_iou (g): {metrics['iou_precision_g']:.4f}")

def print_evaluation_summary(results: List[Dict], total_f1m: float, success_count: int, dataset_config: Dict):
    """Print evaluation summary"""
    if not results:
        print("\nNo evaluation results")
        return
        
    print("\nEvaluation statistics:")
    status_counts = {}
    for result in results:
        if "status" in result:
            status = result["status"]
            if status != "summary":  # Do not count summary items
                status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in status_counts.items():
        print(f"- {status}: {count}")
    print(f"- Total sample count: {len(results) - 1}")  # Do not count summary items
    
    # Get overall metrics from the last result
    if len(results) > 0 and results[-1].get("overall_metrics"):
        overall_metrics = results[-1]["overall_metrics"]
        print("\nOverall evaluation metrics:")
        print(f"- F1M: {overall_metrics['f1m']:.4f}")
        print(f"- F1M_iou (h): {overall_metrics['iou_h']:.4f}")
        print(f"- DSR: {overall_metrics['dsr']:.4f}")
        print(f"\nSample count: {overall_metrics['sample_count']}")

def calculate_iou(span1: Dict[str, int], span2: Dict[str, int]) -> float:
    """Calculate the IOU value of two spans"""
    intersection_start = max(span1['start'], span2['start'])
    intersection_end = min(span1['end'], span2['end'])
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection = intersection_end - intersection_start
    union = (span1['end'] - span1['start']) + (span2['end'] - span2['start']) - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_iou_matches_greedy(pred_spans: List[Dict], gt_spans: List[Dict], iou_threshold: float = 0.5) -> List[tuple]:
    """Calculate the matches between predicted spans and ground truth spans using greedy algorithm
    
    Args:
        pred_spans: List of predicted spans
        gt_spans: List of ground truth spans
        iou_threshold: IOU threshold, default 0.5
        
    Returns:
        List[tuple]: List of (pred_idx, gt_idx) matches
    """
    matches = []
    used_gt = set()
    
    # Sort predicted spans by confidence in descending order
    pred_indices = list(range(len(pred_spans)))
    pred_indices.sort(key=lambda x: pred_spans[x].get('confidence', 0), reverse=True)
    
    # Iterate through predicted spans in descending order of confidence
    for pred_idx in pred_indices:
        pred_span = pred_spans[pred_idx]
        best_iou = iou_threshold
        best_gt_idx = -1
        
        # Find the best match for the ground truth span
        for gt_idx, gt_span in enumerate(gt_spans):
            if gt_idx in used_gt:
                continue
                
            iou = calculate_iou(pred_span, gt_span)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx != -1:
            matches.append((pred_idx, best_gt_idx))
            used_gt.add(best_gt_idx)
    
    return matches

def calculate_iou_matches_hungarian(pred_spans: List[Dict], gt_spans: List[Dict], iou_threshold: float = 0.5) -> List[tuple]:
    """Calculate the matches between predicted spans and ground truth spans using Hungarian algorithm
    
    Args:
        pred_spans: List of predicted spans
        gt_spans: List of ground truth spans
        iou_threshold: IOU threshold, default 0.5
        
    Returns:
        List[tuple]: List of (pred_idx, gt_idx) matches
    """
    if not pred_spans or not gt_spans:
        return []
    
    # Build cost matrix
    cost_matrix = np.zeros((len(pred_spans), len(gt_spans)))
    for i, pred_span in enumerate(pred_spans):
        for j, gt_span in enumerate(gt_spans):
            iou = calculate_iou(pred_span, gt_span)
            # Convert IOU to cost (1-IOU) and handle cases below threshold
            cost_matrix[i][j] = 1000 if iou < iou_threshold else (1 - iou)
    
    # Calculate optimal matches using Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Filter out matches with IOU below threshold
    matches = []
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx][gt_idx] < 1000:  # Only keep valid matches
            matches.append((pred_idx, gt_idx))
    
    return matches

def calculate_iou_precision(pred_spans: List[Dict], gt_spans: List[Dict]) -> float:
    """Calculate F1M_iou"""
    if not pred_spans or not gt_spans:
        return 0.0
    
    total_iou = 0.0
    for pred_span in pred_spans:
        max_iou = 0.0
        for gt_span in gt_spans:
            iou = calculate_iou(pred_span, gt_span)
            max_iou = max(max_iou, iou)
        total_iou += max_iou
    
    return total_iou / len(pred_spans)

def calculate_span_iou(span1: Dict, span2: Dict) -> float:
    """Calculate the IOU (intersection over union) of two text spans
    
    Args:
        span1: First span, contains start and end fields (word count)
        span2: Second span, contains start and end fields (word count)
        
    Returns:
        float: IOU value
    """
    # Calculate intersection
    intersection_start = max(span1['start'], span2['start'])
    intersection_end = min(span1['end'], span2['end'])
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection = intersection_end - intersection_start
    
    # Calculate union
    union = (span1['end'] - span1['start']) + (span2['end'] - span2['start']) - intersection
    
    return intersection / union if union > 0 else 0.0

def non_maximum_suppression(spans: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Use non-maximum suppression to filter out overlapping spans
    
    Args:
        spans: List of predicted spans, each span contains start, end, text and confidence fields
        iou_threshold: IOU threshold, overlapping spans above this value will be suppressed
        
    Returns:
        List[Dict]: List of spans after NMS processing
    """
    if not spans:
        return []
    
    print("\nNMS processing started...")
    print(f"Initial spans number: {len(spans)}")
    
    # Sort by confidence in descending order, for spans with the same confidence, sort by length in descending order (prefer longer spans)
    # spans = sorted(spans, key=lambda x: (x['confidence'], -(x['end'] - x['start'])), reverse=True)
    spans = sorted(spans, key=lambda x: (x['confidence']), reverse=True)
    print("\nSorted spans by confidence and length:")
    for span in spans:
        print(f"Text: {span['text']}, Confidence: {span['confidence']}, Length: {span['end'] - span['start']}, Position: [{span['start']}, {span['end']}]")
    
    # Store spans to be kept
    kept_spans = []
    
    # Iterate through all spans
    while spans:
        current_span = spans[0]
        kept_spans.append(current_span)
        print(f"\nSelected span: {current_span['text']} (Confidence: {current_span['confidence']}, Length: {current_span['end'] - current_span['start']})")
        
        # Remove current span
        spans = spans[1:]
        
        # Filter out spans that overlap with the current span or have an inclusion relationship
        filtered_spans = []
        for span in spans:
            # Check if there is an overlap or inclusion relationship
            has_overlap = False
            
            # Check if there is an overlap
            if (current_span['start'] <= span['end'] and current_span['end'] >= span['start']):
                has_overlap = True
                print(f"Removing overlapping span: {span['text']} (overlapping position)")
                continue
            
            # Check for inclusion relationship
            if (current_span['start'] <= span['start'] and current_span['end'] >= span['end']) or \
               (span['start'] <= current_span['start'] and span['end'] >= current_span['end']):
                has_overlap = True
                print(f"Removing inclusion span: {span['text']} (inclusion relationship)")
                continue
            
            # If there is no overlap and inclusion relationship, keep the span
            if not has_overlap:
                filtered_spans.append(span)
        
        spans = filtered_spans
    
    # Sort by start position
    kept_spans.sort(key=lambda x: x['start'])
    
    print(f"\nNMS processing completed, {len(kept_spans)} spans retained")
    return kept_spans

if __name__ == "__main__":
    test_self_consistency_extraction()