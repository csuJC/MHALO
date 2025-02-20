import os
import base64
from typing import Dict, List, Optional, Union
import re
from openai import OpenAI
import time
from scipy.optimize import linear_sum_assignment
import numpy as np


def extract_hallucination_spans(text: str) -> List[Dict[str, Union[int, str]]]:
    """从文本中提取幻觉片段
    
    Args:
        text (str): 输入文本
        
    Returns:
        list: 包含字典的列表,每个字典包含:
            - start: 起始位置(按单词计数)
            - end: 结束位置(按单词计数)
            - text: 幻觉文本内容
    """
    if not isinstance(text, str):
        print(f"Warning: 输入text不是字符串类型: {type(text)}")
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
    """带重试机制的API调用"""
    for attempt in range(max_retries):
        try:
            print(f"\n尝试调用API (第{attempt + 1}次)...")
            print(f"模型: {model_name}")
            print(f"消息长度: {len(str(messages))} 字符")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1000,
                top_p=0.9,
                temperature=temperature,
                presence_penalty=0.3,
                frequency_penalty=0.3
            )
            
            # 验证响应格式
            if not response:
                raise ValueError("API返回空响应")
                
            if not hasattr(response, 'choices') or not response.choices:
                raise ValueError("API响应中缺少choices字段")
                
            if not hasattr(response.choices[0], 'message'):
                raise ValueError("API响应中缺少message字段")
                
            if not response.choices[0].message.content:
                raise ValueError("API响应中content为空")
                
            print(f"API调用成功!")
            return response
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            print(f"\n第 {attempt + 1} 次调用失败:")
            print(f"错误类型: {error_type}")
            print(f"错误信息: {error_msg}")
            
            if attempt == max_retries - 1:  # 最后一次尝试
                raise Exception(
                    f"API调用失败 (已重试{max_retries}次):\n"
                    f"- 错误类型: {error_type}\n"
                    f"- 错误信息: {error_msg}\n"
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
    elif dataset_type in ["test", "mhal"]:  # test和mhal数据集使用相同的处理逻辑
        # 检查必要字段
        required_fields = ['image_path', 'hallucinated_solution', 'test_solution', 'prompt']
        if not all(field in sample for field in required_fields):
            print(f"样本缺少必要字段: {[f for f in required_fields if f not in sample]}")
            return None
        
        # 构建图片路径
        image_path = os.path.join('images', dataset_type, sample['image_path'])
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return None
        
        # 对于test数据集，使用图片路径和提示词的组合作为ID
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
        # 其他数据集的处理逻辑
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

def normalize_text_for_comparison(text: str) -> str:
    """标准化文本用于比较
    
    Args:
        text: 输入文本
        
    Returns:
        标准化后的文本
    """
    # 1. 移除标点符号
    text = re.sub(r'[^\w\s°]', '', text)
    # 2. 标准化空白字符
    text = ' '.join(text.split())
    return text

def calculate_partial_match_precision(pred_span: Dict, gt_spans: List[Dict]) -> float:
    """计算单个预测跨度的部分匹配精确率，基于单词索引"""
    pred_indices = set(range(pred_span["start"], pred_span["end"]))
    
    if not pred_indices:
        return 0.0
    
    # 标准化预测文本
    pred_text = normalize_text_for_comparison(pred_span["text"])
    pred_words = set(pred_text.split())
    
    if not pred_words:
        return 0.0
    
    max_overlap = 0
    for gt_span in gt_spans:
        # 标准化真实文本
        gt_text = normalize_text_for_comparison(gt_span["text"])
        gt_words = set(gt_text.split())
        
        if not gt_words:
            continue
            
        overlap = len(pred_words & gt_words)
        max_overlap = max(max_overlap, overlap)
    
    return max_overlap / len(pred_words)

def calculate_partial_match_recall(gt_span: Dict, pred_spans: List[Dict]) -> float:
    """计算单个Ground Truth跨度的部分匹配召回率，基于单词索引"""
    gt_indices = set(range(gt_span["start"], gt_span["end"]))
    
    if not gt_indices:
        return 0.0
    
    # 标准化真实文本
    gt_text = normalize_text_for_comparison(gt_span["text"])
    gt_words = set(gt_text.split())
    
    if not gt_words:
        return 0.0
    
    max_overlap = 0
    for pred_span in pred_spans:
        # 标准化预测文本
        pred_text = normalize_text_for_comparison(pred_span["text"])
        pred_words = set(pred_text.split())
        
        if not pred_words:
            continue
            
        overlap = len(gt_words & pred_words)
        max_overlap = max(max_overlap, overlap)
    
    return max_overlap / len(gt_words)

def calculate_f1m_score(ground_truth: str, prediction: str, pred_spans: List[Dict] = None) -> dict:
    """计算F1M分数
    
    Args:
        ground_truth: 真实文本
        prediction: 预测文本
        pred_spans: 预先提取的预测spans(可选)
        
    Returns:
        dict: 包含precision, recall, f1m等指标的字典
    """
    # 提取预测文本中的幻觉片段
    if pred_spans is None:
        pred_spans = extract_hallucination_spans(prediction)
    
    # 提取真实文本中的幻觉片段
    gt_spans = extract_hallucination_spans(ground_truth)
    
    # 计算精确率和召回率
    total_precision = 0
    total_recall = 0
    
    # 计算每个预测span的精确率
    for pred_span in pred_spans:
        precision = calculate_partial_match_precision(pred_span, gt_spans)
        total_precision += precision
    
    # 计算每个真实span的召回率
    for gt_span in gt_spans:
        recall = calculate_partial_match_recall(gt_span, pred_spans)
        total_recall += recall
    
    # 计算最终的精确率和召回率
    precision = total_precision / len(pred_spans) if pred_spans else 0.0
    recall = total_recall / len(gt_spans) if gt_spans else 0.0
    
    # 计算F1分数
    f1m = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 计算IOU精确率
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
    
    # 打印基础指标
    metrics = result['metrics']
    print(f"F1M: {metrics['f1m']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"IOU精确率(h): {metrics['iou_h']:.4f}")
    print(f"IOU精确率(g): {metrics['iou_precision_g']:.4f}")

def print_evaluation_summary(results: List[Dict], total_f1m: float, success_count: int, dataset_config: Dict):
    """打印评估总结"""
    if not results:
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
        print(f"- F1M: {overall_metrics['f1m']:.4f}")
        print(f"- IOU精确率(h): {overall_metrics['iou_h']:.4f}")
        print(f"- DSR: {overall_metrics['dsr']:.4f}")
        print(f"\n样本数量: {overall_metrics['sample_count']}")

def calculate_iou(span1: Dict[str, int], span2: Dict[str, int]) -> float:
    """计算两个span的IOU值"""
    intersection_start = max(span1['start'], span2['start'])
    intersection_end = min(span1['end'], span2['end'])
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection = intersection_end - intersection_start
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

def calculate_iou_precision(pred_spans: List[Dict], gt_spans: List[Dict]) -> float:
    """计算IOU精确率"""
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
    """计算两个文本span的IOU（交并比）
    
    Args:
        span1: 第一个span，包含start和end字段(按单词计数)
        span2: 第二个span，包含start和end字段(按单词计数)
        
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

def non_maximum_suppression(spans: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """对文本spans进行非极大值抑制
    
    Args:
        spans: 包含多个预测span的列表，每个span包含start、end、text和confidence字段
        iou_threshold: IOU阈值，超过此值的重叠span将被抑制
        
    Returns:
        List[Dict]: 经过NMS处理后的spans列表
    """
    if not spans:
        return []
    
    print("\nNMS处理开始...")
    print(f"初始spans数量: {len(spans)}")
    
    # 按置信度降序排序，对于相同置信度的span，按长度降序排序（优先选择较长的span）
    # spans = sorted(spans, key=lambda x: (x['confidence'], -(x['end'] - x['start'])), reverse=True)
    spans = sorted(spans, key=lambda x: (x['confidence']), reverse=True)
    print("\n按置信度和长度排序后的spans:")
    for span in spans:
        print(f"文本: {span['text']}, 置信度: {span['confidence']}, 长度: {span['end'] - span['start']}, 位置: [{span['start']}, {span['end']}]")
    
    # 用于存储保留的spans
    kept_spans = []
    
    # 遍历所有spans
    while spans:
        # 取出置信度最高（或相同置信度中最短）的span
        current_span = spans[0]
        kept_spans.append(current_span)
        print(f"\n选择span: {current_span['text']} (置信度: {current_span['confidence']}, 长度: {current_span['end'] - current_span['start']})")
        
        # 移除当前span
        spans = spans[1:]
        
        # 过滤掉与当前span重叠或存在包含关系的spans
        filtered_spans = []
        for span in spans:
            # 检查是否存在重叠或包含关系
            has_overlap = False
            
            # 检查位置重叠
            if (current_span['start'] <= span['end'] and current_span['end'] >= span['start']):
                has_overlap = True
                print(f"移除重叠span: {span['text']} (位置重叠)")
                continue
            
            # 检查包含关系
            if (current_span['start'] <= span['start'] and current_span['end'] >= span['end']) or \
               (span['start'] <= current_span['start'] and span['end'] >= current_span['end']):
                has_overlap = True
                print(f"移除包含关系span: {span['text']} (包含关系)")
                continue
            
            # 如果没有重叠和包含关系，保留该span
            if not has_overlap:
                filtered_spans.append(span)
        
        spans = filtered_spans
    
    # 按起始位置排序
    kept_spans.sort(key=lambda x: x['start'])
    
    print(f"\nNMS处理完成，保留spans数量: {len(kept_spans)}")
    return kept_spans

if __name__ == "__main__":
    test_self_consistency_extraction()