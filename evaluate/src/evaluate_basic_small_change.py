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
    HALLUCINATION_TYPE_SYSTEM_MESSAGE,
    BASIC_COMPRESSED_SYSTEM_MESSAGE_2_SHOT,
    BASIC_MATH_COMPRESSED_SYSTEM_MESSAGE_2_SHOT,
    MATH_HALLUCINATION_TYPE_SYSTEM_MESSAGE,
    REASON_AND_TAG_SYSTEM_MESSAGE,
    MATH_REASON_AND_TAG_SYSTEM_MESSAGE,
    MATH_SELF_CONSISTENCY_SYSTEM_MESSAGE,
    SELF_CONSISTENCY_SYSTEM_MESSAGE,
    CAPTION_REASON_TAG_SYSTEM_MESSAGE,
    MATH_CAPTION_REASON_TAG_SYSTEM_MESSAGE,
    MATH_PLAN_A_SYSTEM_MESSAGE,
    PLAN_A_SYSTEM_MESSAGE,
    DIRECT_INDEX_SYSTEM_MESSAGE,
    MATH_DIRECT_INDEX_SYSTEM_MESSAGE,
    MATH_CONFIDENCE_SYSTEM_MESSAGE,
    CONFIDENCE_SYSTEM_MESSAGE
)
from utils_basic import (
    encode_image,
    prepare_evaluation_sample,
    calculate_f1m_score,
    calculate_step_f1m_score,
    print_evaluation_result,
    print_evaluation_summary,
    call_api_with_retry,
    calculate_iou_precision,
    extract_hallucination_spans,
    merge_multiple_predictions,
    extract_hallucination_spans_self_consistency,
    extract_hallucination_spans_plan_a,
    extract_hallucination_spans_direct_index
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
# from local_models import LocalModel  # 导入本地模型模块

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# 从环境变量获取API配置
api_key = os.getenv('OPENAI_API_KEY')
api_base = os.getenv('OPENAI_API_BASE')
if not api_key or not api_base:
    raise ValueError("请设置环境变量 OPENAI_API_KEY 和 OPENAI_API_BASE")

client = OpenAI(api_key=api_key, base_url=api_base)

# 添加全局锁用于同步写入
results_lock = Lock()
progress_lock = Lock()
metrics_lock = Lock()

@dataclass
class DatasetConfig:
    name: str  # 数据集名称
    data_path: str  # 数据文件路径
    output_path: str  # 输出文件路径
    system_message: str  # 系统提示词
    user_prompt_template: str  # 用户提示词模板
    tmp_dir: str = ""  # 临时目录路径

def save_results_to_csv(all_results: Dict[str, Dict], model: str, prompt_method: str = "vanilla", output_path: str = "results/tmp/evaluation_summary.csv"):
    """将所有数据集的评估结果保存到CSV文件中
    
    Args:
        all_results: 包含所有数据集评估结果的字典
        model: 使用的模型名称
        prompt_method: 使用的提示词方法
        output_path: CSV文件的输出路径
    """
    # 打印评估配置
    config_df = pd.DataFrame({
        "参数": ["模型", "提示词方法"],
        "值": [model, prompt_method]
    })
    
    print("\n评估配置:")
    print(config_df)
    print("\n" + "="*50 + "\n")
    
    # 准备需要保存的基础指标
    base_metrics = ["f1m", "iou_h", "dsr", "success_count", "total_count"]
    
    # 准备@k的指标
    k_values = [0,1,2, 3, 4, 5, 6, 7, 8, 9]  # 不包括0，因为已经在base_metrics中了
    k_metrics = []
    for k in k_values:
        k_metrics.extend([
            f"f1m@{k}",
            f"iou_h@{k}"
        ])
    
    # 合并所有指标
    metrics = base_metrics + k_metrics
    
    # 准备数据框架
    df = pd.DataFrame(index=metrics)
    
    # 填每个数据集的结果
    for dataset_name, results in all_results.items():
        if results and "overall_metrics" in results[-1]:
            metrics_dict = results[-1]["overall_metrics"]
            # 从summary中获取total_count
            total_count = results[-1].get("total_count", 0)
            
            # 构建结果字典
            result_dict = {
                # 基础指标
                "f1m": metrics_dict.get("f1m", float('nan')),
                "iou_h": metrics_dict.get("iou_h", float('nan')),
                "dsr": metrics_dict.get("dsr", float('nan')),
                "success_count": metrics_dict.get("sample_count", 0),
                "total_count": total_count
            }
            
            # 添加@k的指标
            for k in k_values:
                result_dict[f"f1m@{k}"] = metrics_dict.get(f"f1m@{k}", float('nan'))
                result_dict[f"iou_h@{k}"] = metrics_dict.get(f"iou_h@{k}", float('nan'))
            
            df[dataset_name] = pd.Series(result_dict)
    
    # 计算加权平均值
    success_counts = df.loc['success_count']
    total_success_samples = success_counts.sum()
    total_samples = df.loc['total_count'].sum()
    
    # 计算加权平均值
    weighted_averages = {}
    metrics_to_average = ["f1m", "iou_h"] + [f"f1m@{k}" for k in k_values] + [f"iou_h@{k}" for k in k_values]
    
    for metric in metrics_to_average:
        values = df.loc[metric]
        valid_mask = ~values.isna()
        if valid_mask.any():
            weights = success_counts[valid_mask] / success_counts[valid_mask].sum()
            weighted_avg = (values[valid_mask] * weights).sum()
            weighted_averages[metric] = weighted_avg
        else:
            weighted_averages[metric] = float('nan')
    
    # 添加其他指标
    weighted_averages['dsr'] = total_success_samples / total_samples if total_samples > 0 else 0.0
    weighted_averages['success_count'] = total_success_samples
    weighted_averages['total_count'] = total_samples
    
    # 将加权平均结果添加到数据框中
    df['average'] = pd.Series(weighted_averages)
    
    # 保存到CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将配置信息和评估结果一起保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("评估配置:\n")
        config_df.to_csv(f, index=False)
        f.write("\n评估结果:\n")
        df.to_csv(f)
    
    print(f"\n评估结果已保存到: {output_path}")
    print("\n评估结果汇总:")
    print(df)

def prepare_dataset(dataset, dataset_name: str, sample_limit: Optional[int] = None):
    """准备评测数据集
    
    Args:
        dataset: 原始数据集
        dataset_name: 数据集名称
        sample_limit: 样本数量限制
    
    Returns:
        selected_indices: 选中的样本索引集合
        has_hallucination_indices: 有幻觉的样本索引集合
    """
    if sample_limit is not None and sample_limit % 10 != 0:
        if sample_limit == 2:
            pass
        else:
            raise ValueError("sample_limit必须是10的倍数")
    
    if dataset_name == "test":  # test数据集的特殊处理
        # 不强制比例，直接使用所有样本
        samples_with_hallucination = []
        samples_without_hallucination = []
        
        for idx, sample in enumerate(dataset):
            if sample.get('original_solution') is None:
                samples_with_hallucination.append(idx)
            else:
                samples_without_hallucination.append(idx)
        
        # 对于test数据集，如果sample_limit大于实际样本数，使用所有样本
        all_indices = samples_with_hallucination + samples_without_hallucination
        if sample_limit and sample_limit < len(all_indices):
            selected_indices = set(random.sample(all_indices, sample_limit))
        else:
            print(f"\n注意：test数据集实际样本数({len(all_indices)})小于要求的sample_limit({sample_limit})，将使用全部可用样本。")
            selected_indices = set(all_indices)
        
        # 有幻觉的样本索引就是samples_with_hallucination中的索引
        has_hallucination_indices = set(samples_with_hallucination)
        
    else:  # 其他数据集的处理
        total_samples = sample_limit if sample_limit else len(dataset)
        
        if dataset_name == "mhal":  # mhal数据集的处理
            hallucination_count = int(total_samples * 0.9)
            no_hallucination_count = total_samples - hallucination_count
            
            samples_with_hallucination = []
            samples_without_hallucination = []
            
            for idx, sample in enumerate(dataset):
                if sample.get('original_solution') is None:
                    samples_with_hallucination.append(idx)
                else:
                    samples_without_hallucination.append(idx)
            
            # 随机选择样本
            selected_with_hallucination = set(random.sample(samples_with_hallucination, hallucination_count))
            selected_without_hallucination = set(random.sample(samples_without_hallucination, no_hallucination_count))
            
            selected_indices = selected_with_hallucination | selected_without_hallucination
            has_hallucination_indices = selected_with_hallucination
            
        else:  # 其他数据集的处理
            hallucination_count = int(total_samples * 0.9)
            no_hallucination_count = total_samples - hallucination_count
            
            all_indices = list(range(len(dataset)))
            # 随机选择无幻觉样本的索引
            no_hallucination_indices = set(random.sample(all_indices, no_hallucination_count))
            # 从剩余样本中选择有幻觉样本的索引
            remaining_indices = list(set(all_indices) - no_hallucination_indices)
            has_hallucination_indices = set(random.sample(remaining_indices, hallucination_count))
            
            selected_indices = has_hallucination_indices | no_hallucination_indices
    
    return selected_indices, has_hallucination_indices

def evaluate_single_sample(args):
    """评估单个样本的函数
    
    Args:
        args: 包含评估所需参数的字典
    
    Returns:
        评估结果字典
    """
    sample, idx, dataset_config, has_hallucination_indices, client, progress_bar, model, max_annotation_retries, api_retry_limit, prompt_method = args
    
    try:
        # 准备样本
        result = prepare_evaluation_sample(sample, dataset_config.name)
        if result is None:
            with progress_lock:
                progress_bar.update(1)
            return None
        
        # 设置样本是否有幻觉
        result["has_hallucination"] = idx in has_hallucination_indices
        
        # 对于无幻觉样本的特殊处理
        if not result["has_hallucination"]:
            if result.get('original_solution'):
                result['gt_solution'] = result['original_solution']
                result['test_solution'] = result['original_solution']
            else:
                print(f"警告: 样本 {result['sample_id']} 缺少original_solution字段")
                with progress_lock:
                    progress_bar.update(1)
                return None
        
        # 添加重试计数器
        annotation_retry_count = 0
        while annotation_retry_count < max_annotation_retries:
            # 构建评估提示词
            messages = [
                {
                    "role": "system",
                    "content": dataset_config.system_message
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": dataset_config.user_prompt_template.format(
                                prompt=result['prompt'],
                                test_description=result['test_solution']
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(result['image_path'])}"
                            }
                        }
                    ]
                }
            ]
            
            # 对于self-consistency方法，进行多次预测
            if prompt_method == "self_consistency":
                print(f"\n开始对样本 {result['sample_id']} 进行多次预测...")
                predictions = []
                intermediate_results = {
                    "predictions": [],
                    "final_merged_spans": None,
                    "final_tagged_response": None
                }
                
                # 使用不同的temperature进行多次预测
                temperatures = [0.7, 0.9, 1.1]  # 逐渐增加随机性
                
                for i in range(3):  # 进行3次预测
                    if model == 'tp_model':
                        response = call_tp_model_api(messages)
                    else:
                        response = call_api_with_retry(
                            client, 
                            model, 
                            messages, 
                            max_retries=api_retry_limit, 
                            temperature=temperatures[i]  # 使用不同的temperature
                        )
                        response = response.choices[0].message.content
                    predictions.append(response)
                    print(f"\n第{i+1}次预测结果 (temperature={temperatures[i]}):")
                    print(response)
                    
                    # 提取当前预测的spans
                    current_spans = extract_hallucination_spans_self_consistency(response)
                    print(f"\n第{i+1}次预测提取的幻觉片段:")
                    for span in current_spans:
                        print(f"文本: {span['text']}, 置信度: {span['confidence']}, 位置: [{span['start']}, {span['end']}]")
                    
                    # 保存当前预测的结果
                    intermediate_results["predictions"].append({
                        "iteration": i + 1,
                        "temperature": temperatures[i],
                        "response": response,
                        "extracted_spans": current_spans
                    })
                
                print("\n开始合并多次预测结果...")
                # 合并多次预测结果
                merged_spans = merge_multiple_predictions(predictions)
                
                print("\n合并后的最终幻觉片段:")
                for span in merged_spans:
                    print(f"文本: {span['text']}, 置信度: {span['confidence']}, 位置: [{span['start']}, {span['end']}]")
                
                # 保存合并后的结果
                intermediate_results["final_merged_spans"] = merged_spans
                
                # 构建最终的标注文本
                tagged_response = result['test_solution']
                words = tagged_response.split()
                result_parts = []
                current_pos = 0
                
                # 按位置排序merged_spans
                sorted_spans = sorted(merged_spans, key=lambda x: x['start'])
                
                # 先验证所有spans的文本是否与原文匹配
                for span in sorted_spans:
                    span_text = ' '.join(words[span['start']:span['end']])
                    if span_text != span['text']:
                        print(f"\n警告: span文本与原文不匹配")
                        print(f"原文中的文本: {span_text}")
                        print(f"span中的文本: {span['text']}")
                        # 更新span的文本为原文中的实际文本
                        span['text'] = span_text
                
                # 构建标注文本
                for span in sorted_spans:
                    # 添加span前的文本
                    if current_pos < span['start']:
                        result_parts.append(' '.join(words[current_pos:span['start']]))
                    
                    # 添加带标签的span
                    tag_start = f'<hallucination confidence={span["confidence"]}>'
                    tag_end = '</hallucination>'
                    span_text = ' '.join(words[span['start']:span['end']])
                    result_parts.append(f"{tag_start}{span_text}{tag_end}")
                    
                    current_pos = span['end']
                
                # 添加剩余的文本
                if current_pos < len(words):
                    result_parts.append(' '.join(words[current_pos:]))
                
                tagged_response = ' '.join(result_parts)
                
                # 保存最终的标注文本
                intermediate_results["final_tagged_response"] = tagged_response
                
                print("\n最终标注结果:")
                print(tagged_response)
                
                # 将中间结果保存到主结果中
                result["intermediate_results"] = intermediate_results
                
                # 直接使用merged_spans作为预测结果
                result["pred_spans"] = merged_spans.copy()  # 使用copy避免引用问题
                gpt4v_response = tagged_response
                result["api_original_response"] = "\n".join(predictions)  # 保存所有原始预测
                
            else:
                # 原有的处理逻辑
                if model == 'tp_model':
                    gpt4v_response = call_tp_model_api(messages)
                else:
                    response = call_api_with_retry(client, model, messages, max_retries=api_retry_limit)
                    # 添加响应内容检查和日志
                    # print(f"\n原始API响应:")
                    # print(response)
                    
                    if not response or not hasattr(response, 'choices') or not response.choices:
                        print(f"警告: API响应格式不正确")
                        print(f"响应内容: {response}")
                        raise ValueError("API响应格式不正确")
                        
                    if not hasattr(response.choices[0], 'message') or not response.choices[0].message:
                        print(f"警告: API响应中缺少message字段")
                        print(f"响应内容: {response}")
                        raise ValueError("API响应中缺少message字段")
                        
                    gpt4v_response = response.choices[0].message.content
                    if not gpt4v_response:
                        print(f"警告: API响应中content为空")
                        raise ValueError("API响应中content为空")
                    
                print(f"\n模型原始响应:")  # 添加调试信息
                print(gpt4v_response)
            
            result["api_original_response"] = gpt4v_response
            result["gpt4v_response"] = gpt4v_response
            tagged_response = gpt4v_response
            tmp_response = gpt4v_response
            
            # 对除self-consistency模式外的特殊处理
            if prompt_method == "reason_and_tag" or prompt_method == "vanilla" or prompt_method == "hallucnation_type" or prompt_method == "2-shot" or prompt_method == "caption_reason_tag" or prompt_method == "plan_a" or prompt_method == "direct_index" or prompt_method == "best" or prompt_method == "confidence":
                # 首先提取<Tagged_Text>标签中的内容（不适用于Direct_index模式）
                if prompt_method != "direct_index":
                    tagged_text_pattern = r'Here is the response with hallucinated content tagged:\s*<Tagged_Text>(.*?)</Tagged_Text>'
                    tagged_text_match = re.search(tagged_text_pattern, gpt4v_response, re.DOTALL)
                    if tagged_text_match:
                        tmp_response = tagged_text_match.group(1).strip()
                        tagged_response = tmp_response
                        result["gpt4v_response"] = tmp_response
                        print(f"\n提取的<Tagged_Text>内容:")
                        print(tmp_response)
                        
                        # 为caption_reason_tag和best模式保存额外信息
                        if prompt_method == "caption_reason_tag" or prompt_method == "best":
                            # 提取Caption部分
                            caption_pattern = r'<Caption>(.*?)</Caption>'
                            caption_match = re.search(caption_pattern, gpt4v_response, re.DOTALL)
                            if caption_match:
                                result["caption"] = caption_match.group(1).strip()
                                
                            # 提取Analysis部分
                            analysis_pattern = r'<Analysis>(.*?)</Analysis>'
                            analysis_match = re.search(analysis_pattern, gpt4v_response, re.DOTALL)
                            if analysis_match:
                                result["analysis"] = analysis_match.group(1).strip()
                    else:
                        print(f"警告: 样本 {result['sample_id']} 未找到<Tagged_Text>标签，正在重试...")
                        annotation_retry_count += 1
                        if annotation_retry_count >= max_annotation_retries:
                            print(f"警告: 样本 {result['sample_id']} 在{max_annotation_retries}次尝试后仍未找到<Tagged_Text>标签")
                            result["status"] = "no_tagged_text"
                            with progress_lock:
                                progress_bar.update(1)
                            return result
                        continue
                
                # 为Direct_index模式提取并保存pred_spans
                if prompt_method == "direct_index":
                    print(f"\n原始响应内容:")
                    print(gpt4v_response)
                    pred_spans = extract_hallucination_spans_direct_index(gpt4v_response)
                    result["pred_spans"] = pred_spans
                    print(f"\n提取到的pred_spans: {pred_spans}")
                    # Direct_index模式不需要进行标签检查和比较
                    compare_result = True  # 直接设置为True，跳过标签检查
                    break  # 跳出重试循环
                # 为Plan A模式提取并保存pred_spans
                elif prompt_method == "plan_a":
                    print(f"\n原始响应内容:")
                    print(gpt4v_response)
                    pred_spans = extract_hallucination_spans_plan_a(gpt4v_response)
                    result["pred_spans"] = pred_spans
                    print(f"\n提取到的pred_spans: {pred_spans}")
            
            # 检查标注是否正确（不适用于Direct_index模式）
            if prompt_method != "direct_index":
                if prompt_method == "self_consistency" or prompt_method == "confidence":
                    pattern = r'<hallucination confidence=\d+>(.*?)</hallucination>'
                elif prompt_method == "plan_a":
                    pattern = r'<H>(.*?)</H>'
                else:
                    pattern = r'<hallucination>(.*?)</hallucination>'
                
                # 移除所有标签，比较单词数量
                text_without_tags = re.sub(pattern, r'\1', tmp_response)
                text_without_tags = text_without_tags.strip()
                original_text = result['test_solution'].strip()
                
                # 比较单词数量是否相同
                compare_result = (len(text_without_tags.split()) == len(original_text.split()))
                
                if not compare_result:
                    print(f"警告: 单词数量不匹配")
                    print(f"原始文本单词数: {len(original_text.split())}")
                    print(f"处理后文本单词数: {len(text_without_tags.split())}")
                    print(f"原始文本: {original_text}")
                    print(f"处理后文本: {text_without_tags}")
                    # 打印更多调试信息
                    print(f"\n原始响应内容:")
                    print(gpt4v_response)
                    print(f"\n提取的<Tagged_Text>内容:")
                    print(tmp_response)

            if compare_result:
                break
            
            annotation_retry_count += 1
            if annotation_retry_count < max_annotation_retries:
                print(f"警告: 样本 {result['sample_id']} 第{annotation_retry_count}次标注不正确，正在重试...")
                continue
        
        if not compare_result:
            print(f"警告: 样本 {result['sample_id']} 在{max_annotation_retries}次尝试后仍未正确标注")
            result["status"] = "not_correct_annotated"
            with progress_lock:
                progress_bar.update(1)
            return result
        
        # 提取置信度信息
        confidence_pattern = r'<hallucination confidence=(\d+)>(.*?)</hallucination>'
        confidences = []
        for match in re.finditer(confidence_pattern, tagged_response):
            confidence = int(match.group(1))
            text = match.group(2)
            confidences.append({
                "text": text,
                "confidence": confidence
            })
        result["confidences"] = confidences
        
        # 为confidence模式提取pred_spans
        if prompt_method == "confidence":
            pred_spans = []
            words = result['test_solution'].split()
            current_pos = 0
            
            for confidence_item in confidences:
                # 在原文中查找hallucination文本的位置
                text_words = confidence_item["text"].split()
                text_len = len(text_words)
                
                # 从当前位置开始查找
                found = False
                for i in range(current_pos, len(words) - text_len + 1):
                    if ' '.join(words[i:i+text_len]) == confidence_item["text"]:
                        pred_spans.append({
                            "text": confidence_item["text"],
                            "start": i,
                            "end": i + text_len,
                            "confidence": confidence_item["confidence"]
                        })
                        current_pos = i + text_len
                        found = True
                        break
                
                if not found:
                    print(f"警告: 无法在原文中找到幻觉文本: {confidence_item['text']}")
            
            result["pred_spans"] = pred_spans
            print(f"\n提取到的pred_spans: {pred_spans}")
        
        # 计算不同k值的F1M分数和IOU精确率
        all_metrics_result = {}
        initial_details = None  # 用于保存k=0时的details
        
        for k in [0,1,2, 3,4, 5, 6, 7, 8, 9]:
            if dataset_config.name == "multimath_300k":
                metrics = calculate_step_f1m_score(result["gt_solution"], result["gpt4v_response"], k)
            else:
                # 使用已保存的pred_spans（如果有）
                pred_spans = result.get("pred_spans", [])
                
                # 对于confidence模式，根据置信度阈值k筛选pred_spans
                if prompt_method == "confidence" and k > 0:
                    filtered_pred_spans = [
                        span for span in pred_spans
                        if span["confidence"] >= k
                    ]
                else:
                    filtered_pred_spans = pred_spans
                
                metrics = calculate_f1m_score(
                    result["gt_solution"], 
                    result["gpt4v_response"], 
                    k,
                    pred_spans=filtered_pred_spans
                )
            
            # 保存k=0时的details信息
            if k == 0:
                initial_details = metrics["details"]
            
            # 计算IOU精确率（贪心算法）
            iou_precision_g = calculate_iou_precision(
                metrics["details"]["pred_spans" if dataset_config.name != "multimath_300k" else "pred_steps"],
                metrics["details"]["gt_spans" if dataset_config.name != "multimath_300k" else "gt_steps"],
                k,
                'greedy'
            )
            
            # 计算IOU精确率（匈牙利算法）
            iou_h = calculate_iou_precision(
                metrics["details"]["pred_spans" if dataset_config.name != "multimath_300k" else "pred_steps"],
                metrics["details"]["gt_spans" if dataset_config.name != "multimath_300k" else "gt_steps"],
                k,
                'hungarian'
            )
            
            # 保存到结果中
            prefix = f"@{k}" if k > 0 else ""
            all_metrics_result[f"f1m{prefix}"] = metrics["f1m"]
            all_metrics_result[f"precision{prefix}"] = metrics["precision"]
            all_metrics_result[f"recall{prefix}"] = metrics["recall"]
            all_metrics_result[f"iou_precision_g{prefix}"] = iou_precision_g
            all_metrics_result[f"iou_h{prefix}"] = iou_h
        
        # 使用k=0时的details信息
        result["details"] = initial_details
        result["metrics"] = all_metrics_result
        result["status"] = "success"
        
        # 打印评估结果
        print_evaluation_result(result, dataset_config.name)
        
    except Exception as e:
        print(f"\n样本 {result['sample_id']} 评估失败: {str(e)}")
        result["status"] = "error"
        result["error"] = str(e)
    
    with progress_lock:
        progress_bar.update(1)
    
    return result

def evaluate_dataset(
    dataset_config: DatasetConfig,
    sample_limit: Optional[int] = None,
    restart: bool = True,
    model: str = 'gpt-4o',
    max_workers: int = 4,  # 最大工作线程数
    max_annotation_retries: int = 3,  # 最大标注重试次数
    api_retry_limit: int = 3,  # 最大API重试次数
    prompt_method: str = "vanilla"  # 添加prompt_method参数
):
    """评估数据集（多线程版本）
    
    Args:
        dataset_config: 数据集配置
        sample_limit: 样本数量限制
        restart: 是否重新进行评估
        model: 使用的模型
        max_workers: 最大工作线程数
        max_annotation_retries: 最大标注重试次数
        api_retry_limit: 最大API重试次数
        prompt_method: 使用的提示词方法
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(dataset_config.output_path), exist_ok=True)
    
    # 根据prompt_method选择system_message
    if prompt_method == "hallucnation_type":
        original_system_message = dataset_config.system_message
        # 根据数据集类型选择对应的hallucination_type系统提示词
        if dataset_config.name in ["geo_170k"]:
            dataset_config.system_message = MATH_HALLUCINATION_TYPE_SYSTEM_MESSAGE
        else:
            dataset_config.system_message = HALLUCINATION_TYPE_SYSTEM_MESSAGE
    elif prompt_method == "2-shot":
        original_system_message = dataset_config.system_message
        # 根据数据集类型选择对应的2-shot系统提示词
        if dataset_config.name in ["geo_170k"]:
            dataset_config.system_message = BASIC_MATH_COMPRESSED_SYSTEM_MESSAGE_2_SHOT
        else:
            dataset_config.system_message = BASIC_COMPRESSED_SYSTEM_MESSAGE_2_SHOT
    elif prompt_method == "reason_and_tag":
        original_system_message = dataset_config.system_message
        # 根据数据集类型选择对应的reason_and_tag系统提示词
        if dataset_config.name == "geo_170k":
            dataset_config.system_message = MATH_REASON_AND_TAG_SYSTEM_MESSAGE
        else:
            dataset_config.system_message = REASON_AND_TAG_SYSTEM_MESSAGE
    elif prompt_method == "caption_reason_tag":
        original_system_message = dataset_config.system_message
        # 根据数据集类型选择对应的caption-reason-tag系统提示词
        if dataset_config.name == "geo_170k":
            dataset_config.system_message = MATH_CAPTION_REASON_TAG_SYSTEM_MESSAGE
        else:
            dataset_config.system_message = CAPTION_REASON_TAG_SYSTEM_MESSAGE
    elif prompt_method == "plan_a":  # 添加Plan A模式的处理
        original_system_message = dataset_config.system_message
        # 根据数据集类型选择对应的Plan A系统提示词
        if dataset_config.name == "geo_170k":
            dataset_config.system_message = MATH_PLAN_A_SYSTEM_MESSAGE
        else:
            dataset_config.system_message = PLAN_A_SYSTEM_MESSAGE
    elif prompt_method == "direct_index":  # 添加Direct_index模式的处理
        original_system_message = dataset_config.system_message
        # 根据数据集类型选择对应的Direct_index系统提示词
        if dataset_config.name == "geo_170k":
            dataset_config.system_message = MATH_DIRECT_INDEX_SYSTEM_MESSAGE
        else:
            dataset_config.system_message = DIRECT_INDEX_SYSTEM_MESSAGE
    elif prompt_method == "confidence":  # 添加confidence模式的处理
        original_system_message = dataset_config.system_message
        # 根据数据集类型选择对应的confidence系统提示词
        if dataset_config.name == "geo_170k":
            dataset_config.system_message = MATH_CONFIDENCE_SYSTEM_MESSAGE
        else:
            dataset_config.system_message = CONFIDENCE_SYSTEM_MESSAGE

    # 初始化所有需要的变量
    evaluation_results = []
    all_metrics = {
        k: {"precisions": [], "recalls": [], "f1ms": [], "iou_precisions_g": [], "iou_precisions_h": []}
        for k in [0,1,2, 3,4, 5, 6, 7, 8, 9]
    }
    
    try:
        with open(dataset_config.data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        return
    
    # 准备数据集
    selected_indices, has_hallucination_indices = prepare_dataset(
        dataset, 
        dataset_config.name, 
        sample_limit
    )
    
    # 获取实际的样本总数
    actual_total_samples = len(selected_indices)
    
    # 初始化进度条
    progress_bar = tqdm(total=actual_total_samples, desc=f"评估 {dataset_config.name} 数据集")
    
    # 创建OpenAI客户端
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_API_BASE')
    )

    # 准备评估任务
    evaluation_tasks = [
        (sample, idx, dataset_config, has_hallucination_indices, client, progress_bar, model, max_annotation_retries, api_retry_limit, prompt_method)
        for idx, sample in enumerate(dataset)
        if idx in selected_indices
    ]
    
    # 使用线程池执行评估任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(evaluate_single_sample, task): task
            for task in evaluation_tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_sample):
            result = future.result()
            if result:
                with results_lock:
                    evaluation_results.append(result)
                    # 实时保存结果
                    with open(dataset_config.output_path, 'w', encoding='utf-8') as f:
                        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    progress_bar.close()
    
    # 计算整体指标
    success_count = sum(1 for result in evaluation_results if result.get("status") == "success")
    overall_metrics = {}
    
    # 计算DSR (Detection Success Rate)
    total_count = actual_total_samples  # 使用实际的样本总数
    dsr = success_count / total_count if total_count > 0 else 0.0
    overall_metrics["dsr"] = dsr
    
    for k in all_metrics.keys():
        metrics = {"precisions": [], "recalls": [], "f1ms": [], "iou_precisions_g": [], "iou_precisions_h": []}
        for result in evaluation_results:
            if result.get("status") == "success" and result.get("metrics"):
                prefix = f"@{k}" if k > 0 else ""
                metrics["precisions"].append(result["metrics"][f"precision{prefix}"])
                metrics["recalls"].append(result["metrics"][f"recall{prefix}"])
                metrics["f1ms"].append(result["metrics"][f"f1m{prefix}"])
                metrics["iou_precisions_g"].append(result["metrics"][f"iou_precision_g{prefix}"])
                metrics["iou_precisions_h"].append(result["metrics"][f"iou_h{prefix}"])
        
        if metrics["precisions"]:
            prefix = f"@{k}" if k > 0 else ""
            overall_metrics[f"f1m{prefix}"] = sum(metrics["f1ms"]) / len(metrics["f1ms"])
            overall_metrics[f"precision{prefix}"] = sum(metrics["precisions"]) / len(metrics["precisions"])
            overall_metrics[f"recall{prefix}"] = sum(metrics["recalls"]) / len(metrics["recalls"])
            overall_metrics[f"iou_precision_g{prefix}"] = sum(metrics["iou_precisions_g"]) / len(metrics["iou_precisions_g"])
            overall_metrics[f"iou_h{prefix}"] = sum(metrics["iou_precisions_h"]) / len(metrics["iou_precisions_h"])
    
    overall_metrics["sample_count"] = success_count
    
    # 添加整体指标到结果中
    evaluation_results.append({
        "status": "summary",
        "overall_metrics": overall_metrics,
        "total_count": total_count  # 添加实际的样本总数到summary中
    })
    
    # 保存最终结果
    with open(dataset_config.output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    # 打印评估总结
    print_evaluation_summary(evaluation_results, sum(metrics["f1ms"]), success_count, dataset_config)

def evaluate_all_datasets(sample_limit: Optional[int] = None, restart: bool = True, model: str = 'gpt-4o', prompt_method: str = "vanilla", max_workers: int = 4, max_annotation_retries: int = 3, api_retry_limit: int = 3):
    """评估所有数据集

    Args:
        sample_limit: 样本数量限制
        restart: 是否重新进行评估
        model: 使用的模型
        prompt_method: 使用的提示词方法
        max_workers: 最大工作线程数
        max_annotation_retries: 最大标注重试次数
        api_retry_limit: 最大API重试次数
    """
    all_results = {}
    print("评估所有数据集")
    
    # 处理模型名称中的特殊字符
    safe_model_name = model.replace('/', '_').replace('\\', '_')
    
    # 生成临时目录名称
    date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tmp_dir_name = f"tmp_{date_time}_{safe_model_name}_{prompt_method}_{sample_limit}"
    tmp_dir = os.path.join("results", tmp_dir_name)
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 更新每个数据集的临时输出路径
    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        dataset_config.tmp_dir = tmp_dir
        dataset_config.output_path = os.path.join(tmp_dir, f"evaluation_results_{dataset_name}.json")
        
        # 根据prompt_method选择system_message
        if prompt_method == "self_consistency":
            original_system_message = dataset_config.system_message
            # 根据数据集类型选择对应的self-consistency系统提示词
            if dataset_name in ["geo_170k"]:
                dataset_config.system_message = MATH_SELF_CONSISTENCY_SYSTEM_MESSAGE
            else:
                dataset_config.system_message = SELF_CONSISTENCY_SYSTEM_MESSAGE
    
    # 评估过程
    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        print(f"\n开始评估 {dataset_name.upper()} 数据集...")
        try:
            # 评估单个数据集
            evaluate_dataset(
                dataset_config=dataset_config,
                sample_limit=sample_limit,
                restart=restart,
                model=model,
                max_workers=max_workers,
                max_annotation_retries=max_annotation_retries,
                api_retry_limit=api_retry_limit,
                prompt_method=prompt_method
            )
            
            # 读取评估结果
            with open(dataset_config.output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_results[dataset_name] = results
                
        except Exception as e:
            print(f"评估数据集 {dataset_name} 时发生错误: {str(e)}")
    
    # 保存汇总结果到CSV
    save_results_to_csv(all_results, model=model, prompt_method=prompt_method, output_path=os.path.join(tmp_dir, "evaluation_summary.csv"))
    
    # 创建最终结果目录
    final_dir_name = f"{date_time}_{safe_model_name}_{prompt_method}_{sample_limit}"
    final_dir = os.path.join("results", final_dir_name)
    os.makedirs(final_dir, exist_ok=True)
    
    # 将临时目录中的所有文件复制到最终目录
    for file in os.listdir(tmp_dir):
        shutil.copy(os.path.join(tmp_dir, file), os.path.join(final_dir, file))
    
    # 删除临时目录
    shutil.rmtree(tmp_dir)
    
    print(f"\n评估结果已保存到: {final_dir}")

def call_tp_model_api(messages):
    """调用TP模型API"""
    url = 'http://10.181.13.226:5000/v1/chat/completions'
    # 获取用户消息中的图片和文本
    for msg in messages:
        if msg["role"] == "user":
            for item in msg["content"]:
                if item["type"] == "image_url":
                    image_path = item["image_url"]["url"]
                else:
                    prompt = item["text"]
        if msg["role"] == "system":
            system_message = msg["content"]

    print("input to api:")
    print(system_message + "\n\n" + prompt)
    
    data = {
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_path}
                },
                {
                    "type": "text", 
                    "text": system_message + prompt
                }
            ]
        }],
        "model": "tp_model",
        "tools": [],
        "stream": False,
        "do_sample": False,
        "best_of": 1,
        "top_p": 0.0001,
        "top_k": 1,
        "temperature": 0.8,
        "repetition_penalty": 1.1,
        "decoder_input_details": False
    }
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    
    if response.status_code != 200:
        return "<ERROR>"
    
    return response.json()['choices'][0]['message']['content']

def main():
    parser = argparse.ArgumentParser(description='评估MLLM模型的表现')
    parser.add_argument('--dataset', type=str, default='all',
                      choices=['all'] + list(DATASET_CONFIGS.keys()),
                      help='要评估的数据集，默认为"all"表示评估所有数据集')
    parser.add_argument('--sample_limit', type=int, default=50,
                      help='评估样本数量限制(须是10的倍数)')
    parser.add_argument('--restart', type=bool, default=True,
                      help='是否重新进行评估。如果为False，则从已有结果文件中读取数据')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20',
                      choices=['gpt-4o-2024-11-20',
                               'claude-3-5-sonnet-20241022', 
                               'gemini-1.5-pro-002', 
                               'qwen-vl-max-0809', 
                               'tp_model',
                               'abab7-chat-preview',
                               'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
                               'glm-4v',
                               'local/MiniCPM-V-2_6',
                               'local/InternVL2-Llama3-76B'],
                      help='选择用的模型，默认为gpt-4o-2024-11-20')
    parser.add_argument('--prompt_method', type=str, default='vanilla',
                      choices=['vanilla', 'hallucnation_type', '2-shot', 'reason_and_tag', 
                              'self_consistency', 'caption_reason_tag', 'plan_a', 'direct_index', 'best', 'confidence'],
                      help='选择使用的提示词方法，默认为vanilla')
    parser.add_argument('--max_workers', type=int, default=16,
                      help='最大工作线程数，默认为16')
    parser.add_argument('--max_annotation_retries', type=int, default=3,
                      help='最大标注重试次数，默认为3')
    parser.add_argument('--api_retry_limit', type=int, default=3,
                      help='最大API调用重试次数，默认为3')
    args = parser.parse_args()
    
    # 检查环境变量
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("请设置环境变量 OPENAI_API_KEY")
    if not os.getenv('OPENAI_API_BASE'):
        raise ValueError("请设置环境变量 OPENAI_API_BASE")
    
    # 验证sample_limit
    if args.sample_limit is not None and args.sample_limit % 10 != 0:
        if args.sample_limit == 2:
            pass
        else:       
            raise ValueError("sample_limit必须是10的倍数")
    
    if args.dataset == 'all':
        # 评估所有数据集
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
        # 评估单个数据集
        dataset_config = DATASET_CONFIGS[args.dataset]
        
        # 根据prompt_method选择system_message
        if args.prompt_method == "self_consistency":
            original_system_message = dataset_config.system_message
            # 根据数据集类型选择对应的self-consistency系统提示词
            if args.dataset in ["geo_170k"]:
                dataset_config.system_message = MATH_SELF_CONSISTENCY_SYSTEM_MESSAGE
            else:
                dataset_config.system_message = SELF_CONSISTENCY_SYSTEM_MESSAGE
        
        print(f"\n开始评估 {args.dataset.upper()} 数据集...")
        evaluate_dataset(
            dataset_config=dataset_config,
            sample_limit=args.sample_limit,
            restart=args.restart,
            model=args.model,
            max_workers=args.max_workers,
            max_annotation_retries=args.max_annotation_retries,
            api_retry_limit=args.api_retry_limit,
            prompt_method=args.prompt_method
        )

if __name__ == "__main__":
    main()