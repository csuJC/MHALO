import os
import base64
from typing import Dict, List, Optional, Union
import re
from openai import OpenAI
import time
from scipy.optimize import linear_sum_assignment
import numpy as np

def call_api_with_retry(client, model_name, messages, max_retries=3, retry_delay=5):
    """带重试机制的API调用"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0
            )
            return response
        except Exception as e:
            if "502" in str(e):
                error_msg = (
                    f"\n502 Bad Gateway错误:"
                    f"\n- 错误详情: {str(e)}"
                    f"\n- 模型: {model_name}"
                    f"\n- 重试次数: {attempt + 1}/{max_retries}"
                    f"\n- 请求内容长度: {len(str(messages))} 字符"
                )
                print(error_msg)
            else:
                print(f"\n遇到错误: {str(e)}")
                
            if attempt == max_retries - 1:  # 最后一次尝试
                raise Exception(
                    f"API调用失败 (已重试{max_retries}次):\n"
                    f"- 错误类型: {'502 Bad Gateway' if '502' in str(e) else str(e)}\n"
                    f"- 模型: {model_name}\n"
                    f"- 请求内容长度: {len(str(messages))} 字符"
                )
            
            print(f"{retry_delay}秒后进行第{attempt + 2}次尝试...")
            time.sleep(retry_delay)
            continue


def encode_image(image_path: str) -> str:
    """将图片编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_evaluation_sample(sample: Dict, dataset_type: str) -> Optional[Dict]:
    """准备评估样本"""
    if dataset_type in ["geo_170k", "mathv_360k", "multimath_300k"]:  # 数学题数据集的处理逻辑
        # 检查数学题数据集必要字段
        required_fields = ['image', 'hallucinated_solution', 'test_solution']
        if dataset_type == "multimath_300k":
            required_fields.append('title')  # multimath_300k 使用 title
        else:
            required_fields.append('question')  # 其他数据集使用 question
            
        if not all(field in sample for field in required_fields):
            print(f"样本缺少必要字段: {[f for f in required_fields if f not in sample]}")
            return None
        
        # 构建图片路径
        image_path = os.path.join('images', dataset_type, sample['image'])
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return None
        
        # 对于multimath_300k数据集的特殊处理
        if dataset_type == "multimath_300k":
            return {
                "sample_id": sample['image'],  # 使用image路径作为id
                "image_path": image_path,
                "original_solution": sample['original_solution'] if isinstance(sample, dict) else None,
                "gt_solution": sample['hallucinated_solution'],
                "test_solution": sample['test_solution'],
                "prompt": sample['title']  # multimath_300k 使用 title
            }
        else:
            return {
                "sample_id": sample['image'],  # 使用image路径作为id
                "image_path": image_path,
                "original_solution": sample.get('original_solution'),
                "gt_solution": sample['hallucinated_solution'],
                "test_solution": sample['test_solution'],
                "prompt": sample['question']  # 其他数据集使用 question
            }
    else:
        # 原有数据集的处理逻辑保持不变
        required_fields = ['id', 'image_path', 'hallucinated_solution', 'prompt']
        if not all(field in sample for field in required_fields):
            print(f"样本缺少必要字段: {[f for f in required_fields if f not in sample]}")
            return None
        
        # 构建图片路径
        image_path = os.path.join('images', dataset_type, sample['image_path'])
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return None
        
        return {
            "sample_id": sample['id'],
            "image_path": image_path,
            "original_solution": sample.get('original_solution'),
            "gt_solution": sample['hallucinated_solution'],
            "test_solution": sample['test_solution'],
            "prompt": sample['prompt']
        }

def extract_hallucination_spans(text: str) -> List[Dict[str, Union[int, str]]]:
    """
    从文本中提取幻觉片段,返回每个片段的起始位置、结束位置、文本内容和confidence值(如果有)
    
    Args:
        text (str): 输入文本
        
    Returns:
        list: 包含字典的列表,每个字典包含:
            - start: 起始位置(按单词计数)
            - end: 结束位置(按单词计数)
            - text: 幻觉文本内容
            - confidence: confidence值(如果有)
    """
    if not isinstance(text, str):
        print(f"Warning: 输入text不是字符串类型: {type(text)}")
        return []

    # 定义两种模式的正则表达式
    pattern_with_conf = r'<hallucination confidence=(\d+)>(.*?)</hallucination>'
    pattern_without_conf = r'<hallucination>(.*?)</hallucination>'
    
    # 存储所有单词的列表
    words = text.split()
    
    # 存储结果
    spans = []
    
    # 首先处理带confidence的情况
    for match in re.finditer(pattern_with_conf, text):
        confidence = int(match.group(1))
        hallucination_text = match.group(2)
        
        # 计算这段幻觉文本在原文中的起始和结束位置(按单词)
        start_char = match.start()
        prefix_text = text[:start_char]
        start_word = len(prefix_text.split())
        
        # 计算结束位置
        hallucination_words = hallucination_text.split()
        end_word = start_word + len(hallucination_words)
        
        spans.append({
            'start': start_word,
            'end': end_word,
            'text': hallucination_text,
            'confidence': confidence
        })
    
    # 处理不带confidence的情况
    for match in re.finditer(pattern_without_conf, text):
        hallucination_text = match.group(1)
        
        # 计算这段幻觉文本在原文中的起始和结束位置(按单词)
        start_char = match.start()
        prefix_text = text[:start_char]
        start_word = len(prefix_text.split())
        
        # 计算结束位置
        hallucination_words = hallucination_text.split()
        end_word = start_word + len(hallucination_words)
        
        spans.append({
            'start': start_word,
            'end': end_word,
            'text': hallucination_text,
            'confidence': 0  # 对于不带confidence的情况，设置为0
        })
    
    # 按起始位置排序
    spans.sort(key=lambda x: x['start'])
    
    return spans

def calculate_partial_match_recall(gt_span: Dict, pred_spans: List[Dict]) -> float:
    """计算单个Ground Truth跨度的部分匹配召回率"""
    gt_text = gt_span["text"].lower()
    gt_words = set(gt_text.split())
    
    if not gt_words:
        return 0.0
    
    max_overlap = 0
    for pred_span in pred_spans:
        pred_text = pred_span["text"].lower()
        pred_words = set(pred_text.split())
        
        overlap = len(gt_words & pred_words)
        max_overlap = max(max_overlap, overlap)
    
    return max_overlap / len(gt_words)

def calculate_partial_match_precision(pred_span: Dict, gt_spans: List[Dict]) -> float:
    """计算单个预测跨度的部分匹配精确率"""
    pred_text = pred_span["text"].lower()
    pred_words = set(pred_text.split())
    
    if not pred_words:
        return 0.0
    
    max_overlap = 0
    for gt_span in gt_spans:
        gt_text = gt_span["text"].lower()
        gt_words = set(gt_text.split())
        
        overlap = len(pred_words & gt_words)
        max_overlap = max(max_overlap, overlap)
    
    return max_overlap / len(pred_words)

def calculate_f1m_score(ground_truth: str, prediction: str, confidence_threshold: int = 0) -> dict:
    """计算F1M分数
    
    Args:
        ground_truth: 真实解答(带有<hallucination>标签)
        prediction: 预测解答(带有<hallucination confidence=X>标签)
        confidence_threshold: 置信度阈值，只考虑confidence大于此值的span
        
    Returns:
        包含F1M分数和其他指标的字典
    """
    # 幻觉片段
    gt_spans = extract_hallucination_spans(ground_truth)
    pred_spans = [span for span in extract_hallucination_spans(prediction) 
                 if span.get('confidence', 0) > confidence_threshold]
    
    # 如果没有预测跨度和真实跨度，返回None
    if not gt_spans and not pred_spans:
        return {
            "f1m": 1.0,  # 完全正确
            "precision": 1.0,
            "recall": 1.0,
            "details": {
                "gt_spans": gt_spans,
                "pred_spans": pred_spans
            }
        }
    elif not gt_spans:  # 有预测跨度但没有真实跨度
        return {
            "f1m": 0.0,  # 完全错误
            "precision": 0.0,
            "recall": 1.0,
            "details": {
                "gt_spans": gt_spans,
                "pred_spans": pred_spans
            }
        }
    elif not pred_spans:  # 有真实跨度但没有预测跨度
        return {
            "f1m": 0.0,  # 完全错误
            "precision": 1.0,
            "recall": 0.0,
            "details": {
                "gt_spans": gt_spans,
                "pred_spans": pred_spans
            }
        }
    
    # 计算每个预测跨度的精确率
    precisions = []
    for pred_span in pred_spans:
        precision = calculate_partial_match_precision(pred_span, gt_spans)
        precisions.append(precision)
    
    # 计算每个真实跨度的召回率
    recalls = []
    for gt_span in gt_spans:
        recall = calculate_partial_match_recall(gt_span, pred_spans)
        recalls.append(recall)
    
    # 计算平均精确率和召回率
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    
    # 计算F1M分数
    if avg_precision + avg_recall == 0:
        f1m = 0.0
    else:
        f1m = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    
    return {
        "f1m": f1m,
        "precision": avg_precision,
        "recall": avg_recall,
        "details": {
            "gt_spans": gt_spans,
            "pred_spans": pred_spans
        }
    }

def extract_step_hallucinations(text: str) -> List[Dict[str, Union[int, str]]]:
    """从标签的文本中提取步骤级别的幻觉
    
    Args:
        text: 带有<hallucination confidence=X>标签的文本
        
    Returns:
        包含每个幻觉步骤信息的列表
    """
    spans = []
    # 修改正则表式以只匹配Step N部分
    pattern = r'<hallucination confidence=(\d+)>(Step \d+)</hallucination>'
    
    for match in re.finditer(pattern, text):
        confidence = int(match.group(1))
        step_text = match.group(2)
        
        # 提取步骤号
        step_num = int(re.search(r'Step (\d+)', step_text).group(1))
        
        # 获取完整的步骤内容
        full_step_pattern = f"{step_text}[^S]*?"  # 匹配到下一个Step之前的内容
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
    """计算步骤级别的F1M分数
    
    Args:
        gt_solution: 真实解答(带有<hallucination>标签)
        pred_solution: 预测解答(带有<hallucination confidence=X>标签)
        confidence_threshold: 置信度阈值，只考虑confidence大于此值的span
        
    Returns:
        包含F1M分数和其他指标的字典
    """
    # 提取实幻觉步骤
    gt_pattern = r'<hallucination>(Step \d+)</hallucination>'
    gt_steps = []
    for match in re.finditer(gt_pattern, gt_solution):
        step_text = match.group(1)
        # 获取完整的步骤内容
        full_step_pattern = f"{step_text}[^S]*?"
        full_step_match = re.search(full_step_pattern, gt_solution)
        if full_step_match:
            gt_steps.append(full_step_match.group(0).strip())
        else:
            gt_steps.append(step_text)
    gt_steps = set(gt_steps)
    
    # 提取预测幻觉步骤，并根据置信度阈值过滤
    pred_steps = set()
    for step in extract_step_hallucinations(pred_solution):
        if step.get('confidence', 0) > confidence_threshold:
            pred_steps.add(step["text"])
    
    # 计算精确率和召回率
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
    
    # 计算交集
    correct_steps = gt_steps & pred_steps
    
    precision = len(correct_steps) / len(pred_steps)
    recall = len(correct_steps) / len(gt_steps)
    
    # 计算F1M
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
    """打印单个样本的评估结果"""
    print(f"\n样本 {result['sample_id']} 评估结果:")
    print(f"Ground Truth: {result['gt_solution']}")
    print(f"GPT-4V标注: {result['gpt4v_response']}")
    
    # 打印所有k值的指标
    metrics = result['metrics']
    for k in sorted([int(key.split('@')[1]) if '@' in key else 0 for key in metrics.keys() if key.startswith('f1m')]):
        prefix = f"@{k}" if k > 0 else ""
        print(f"F1M{prefix}: {metrics[f'f1m{prefix}']:.4f}")
        print(f"精确率{prefix}: {metrics[f'precision{prefix}']:.4f}")
        print(f"召回率{prefix}: {metrics[f'recall{prefix}']:.4f}")

def print_evaluation_summary(results: List[Dict], total_f1m: float, success_count: int, dataset_config: Dict):
    """打印评估总结"""
    if not results:  # 添加列表检查
        print("\n没有评估结果")
        return
        
    print("\n评估统计:")
    status_counts = {}
    for result in results:
        if "status" in result:
            status = result["status"]
            if status != "summary":  # 不统计summary条目
                status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in status_counts.items():
        print(f"- {status}: {count}")
    print(f"- 总样本数: {len(results) - 1}")  # 减去summary条目
    
    # 获取最后一个结果中的整体指标
    if len(results) > 0 and results[-1].get("overall_metrics"):
        overall_metrics = results[-1]["overall_metrics"]
        print("\n整体评估指标:")
        
        # 打印所有k值的指标
        k_values = sorted([int(key.split('@')[1]) if '@' in key else 0 
                         for key in overall_metrics.keys() 
                         if key.startswith('f1m')])
        
        for k in k_values:
            prefix = f"@{k}" if k > 0 else ""
            print(f"\nF1M{prefix}:")
            print(f"- F1M: {overall_metrics[f'f1m{prefix}']:.4f}")
            print(f"- Precision: {overall_metrics[f'precision{prefix}']:.4f}")
            print(f"- Recall: {overall_metrics[f'recall{prefix}']:.4f}")
        
        print(f"\n样本数量: {overall_metrics['sample_count']}")

def calculate_iou(span1: Dict, span2: Dict) -> float:
    """计算两个span的IOU（交并比）
    
    Args:
        span1: 第一个span，包含start和end字段
        span2: 第二个span，包含start和end字段
        
    Returns:
        float: IOU值
    """
    # 计算交集
    intersection_start = max(span1['start'], span2['start'])
    intersection_end = min(span1['end'], span2['end'])
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection = intersection_end - intersection_start
    
    # 计算并集
    union = (span1['end'] - span1['start']) + (span2['end'] - span2['start']) - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_iou_matches_greedy(pred_spans: List[Dict], gt_spans: List[Dict], iou_threshold: float = 0.5) -> List[tuple]:
    """使用贪心算法计算预测spans和真实spans的匹配
    
    Args:
        pred_spans: 预测的spans列表
        gt_spans: 真实的spans列表
        iou_threshold: IOU阈值，默认0.5
        
    Returns:
        List[tuple]: 匹配的(pred_idx, gt_idx)列表
    """
    matches = []
    used_gt = set()
    
    # 按置信度降序排序预测spans的索引
    pred_indices = list(range(len(pred_spans)))
    pred_indices.sort(key=lambda x: pred_spans[x].get('confidence', 0), reverse=True)
    
    # 按置信度顺序遍历预测框
    for pred_idx in pred_indices:
        pred_span = pred_spans[pred_idx]
        best_iou = iou_threshold
        best_gt_idx = -1
        
        # 寻找最佳匹配的真实框
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
    """使用匈牙利算法计算预测spans和真实spans的匹配
    
    Args:
        pred_spans: 预测的spans列表
        gt_spans: 真实的spans列表
        iou_threshold: IOU阈值，默认0.5
        
    Returns:
        List[tuple]: 匹配的(pred_idx, gt_idx)列表
    """
    if not pred_spans or not gt_spans:
        return []
    
    # 构建成本矩阵
    cost_matrix = np.zeros((len(pred_spans), len(gt_spans)))
    for i, pred_span in enumerate(pred_spans):
        for j, gt_span in enumerate(gt_spans):
            iou = calculate_iou(pred_span, gt_span)
            # 将IOU转换为成本（1-IOU），并处理低于阈值的情况
            cost_matrix[i][j] = 1000 if iou < iou_threshold else (1 - iou)
    
    # 使用匈牙利算法计算最优匹配
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # 过滤掉IOU低于阈值的匹配
    matches = []
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx][gt_idx] < 1000:  # 只保留有效匹配
            matches.append((pred_idx, gt_idx))
    
    return matches

def calculate_iou_precision(pred_spans: List[Dict], gt_spans: List[Dict], 
                          confidence_threshold: int = 0, method: str = 'greedy') -> float:
    """计算IOU精确率
    
    Args:
        pred_spans: 预测的spans列表
        gt_spans: 真实的spans列表
        confidence_threshold: 置信度阈值
        method: 使用的匹配方法，'greedy'或'hungarian'
        
    Returns:
        float: IOU精确率
    """
    # 确保输入是列表
    if not isinstance(pred_spans, list) or not isinstance(gt_spans, list):
        print(f"Warning: 输入spans类型错误 - pred_spans: {type(pred_spans)}, gt_spans: {type(gt_spans)}")
        return 0.0

    # 如果spans是字符串，先提取hallucination spans
    if pred_spans and isinstance(pred_spans[0], str):
        pred_spans = extract_hallucination_spans(pred_spans[0])
    if gt_spans and isinstance(gt_spans[0], str):
        gt_spans = extract_hallucination_spans(gt_spans[0])
    
    # 根据置信度阈值过滤预测spans
    filtered_pred_spans = [span for span in pred_spans 
                         if isinstance(span, dict) and span.get('confidence', 0) > confidence_threshold]
    
    if not filtered_pred_spans:
        return 0.0
    
    # 根据选择的方法计算匹配
    if method == 'greedy':
        matches = calculate_iou_matches_greedy(filtered_pred_spans, gt_spans)
    else:  # hungarian
        matches = calculate_iou_matches_hungarian(filtered_pred_spans, gt_spans)
    
    # 计算精确率
    return len(matches) / len(filtered_pred_spans) if filtered_pred_spans else 0.0