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
    MATH_REASON_AND_TAG_SYSTEM_MESSAGE
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

def save_results_to_csv(all_results: Dict[str, List[Dict]], model: str, prompt_method: str = "vanilla", output_path: str = "results/tmp/evaluation_summary.csv"):
    """将所有数据集的评估结果保存到CSV文件中"""
    
    # 计算每个数据集的指标
    dataset_metrics = {}
    total_samples = 0
    total_success = 0
    weighted_f1m_sum = 0
    weighted_iou_h_sum = 0
    
    # 收集每个数据集的指标
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
                    metrics = item.get('metrics', {})
                    f1m_values.append(metrics.get('f1m', 0.0))
                    iou_h_values.append(metrics.get('iou_h', 0.0))
                dataset_total += 1
        
        if dataset_total > 0:
            avg_f1m = sum(f1m_values) / dataset_total
            avg_iou_h = sum(iou_h_values) / dataset_total
            
            dataset_metrics[dataset_name] = {
                'f1m': round(avg_f1m, 4),
                'iou_h': round(avg_iou_h, 4),
                'dsr': round(dataset_success / dataset_total, 4),
                'success_count': dataset_success,
                'total_count': dataset_total
            }
            
            weighted_f1m_sum += avg_f1m * dataset_total
            weighted_iou_h_sum += avg_iou_h * dataset_total
            total_samples += dataset_total
            total_success += dataset_success
    
    # 计算总体加权平均值
    weighted_f1m = weighted_f1m_sum / total_samples if total_samples > 0 else 0
    weighted_iou_h = weighted_iou_h_sum / total_samples if total_samples > 0 else 0
    dsr = total_success / total_samples if total_samples > 0 else 0
    
    # 创建评估配置部分
    config_data = {
        '参数': ['模型', '提示词方法'],
        '值': [model, prompt_method]
    }
    config_df = pd.DataFrame(config_data)
    
    # 创建评估结果部分，包含每个数据集的结果和总体平均值
    results_data = {
        '': ['f1m', 'iou_h', 'dsr', 'success_count', 'total_count']
    }
    
    # 添加每个数据集的结果
    for dataset_name, metrics in dataset_metrics.items():
        # 如果是test数据集,保存时改名为MC
        save_name = "MC" if dataset_name == "test" else dataset_name
        results_data[save_name] = [
            metrics['f1m'],
            metrics['iou_h'],
            metrics['dsr'],
            metrics['success_count'],
            metrics['total_count']
        ]
    
    # 添加总体平均值
    results_data['average'] = [
        round(weighted_f1m, 4),
        round(weighted_iou_h, 4),
        round(dsr, 4),
        total_success,
        total_samples
    ]
    
    results_df = pd.DataFrame(results_data)
    
    # 保存到CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('评估配置:\n')
        config_df.to_csv(f, index=False)
        f.write('\n评估结果:\n')
        results_df.to_csv(f, index=False)
    
    print(f"\n评估结果已保存到: {output_path}")

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
            
            # 修改API调用部分
            response = call_api_with_retry(client, model, messages, max_retries=api_retry_limit)
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
            
            print(f"\n模型原始响应:")
            print(gpt4v_response)
            
            result["api_original_response"] = gpt4v_response
            result["gpt4v_response"] = gpt4v_response
            tagged_response = gpt4v_response
            tmp_response = gpt4v_response
            
            # 对除self-consistency模式外的特殊处理
            if (prompt_method == "Analyze-then-judge" or 
                prompt_method == "vanilla" or 
                prompt_method == "Criteria" or 
                prompt_method == "2-shot"):
                # 提取<Tagged_Text>标签中的内容
                tagged_text_pattern = r'<Tagged_Text>(.*?)</Tagged_Text>'
                tagged_text_match = re.search(tagged_text_pattern, tmp_response, re.DOTALL)
                if tagged_text_match:
                    tmp_response = tagged_text_match.group(1).strip()
                    tagged_response = tmp_response
                    gpt4v_response = tmp_response
                    result["gpt4v_response"] = gpt4v_response
                    print(f"\n提取的<Tagged_Text>内容:")
                    print(gpt4v_response)
                    
            # 检查标注是否正确（不适用于Direct_index模式）
            if prompt_method != "direct_index":
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
        
        # 计算不同k值的F1M分数和IOU精确率
        metrics = calculate_f1m_score(
            result["gt_solution"], 
            result["gpt4v_response"],
            pred_spans=result.get("pred_spans")
        )
        
        result["metrics"] = metrics
        result["details"] = metrics["details"]
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
    model: str = "gpt-4o-2024-11-20",
    max_workers: int = 16,
    max_annotation_retries: int = 3,
    api_retry_limit: int = 3,
    prompt_method: str = "vanilla",
    return_results: bool = False,
    result_dir: Optional[str] = None  # 新增参数
) -> Optional[List[Dict]]:
    """评估单个数据集"""
    
    # 如果没有提供result_dir，则创建新的时间戳目录
    if result_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        safe_model_name = model.replace('/', '_')
        result_dir = f"results/{timestamp}_{safe_model_name}_{prompt_method}_{sample_limit}"
    
    # 在共享目录下创建数据集子目录
    dataset_dir = os.path.join(result_dir, dataset_config.name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 修改详细结果的保存路径
    detailed_output_path = os.path.join(dataset_dir, f"{dataset_config.name}_detailed_results.json")
    
    # 加载数据集
    with open(dataset_config.data_path, 'r') as f:
        dataset = json.load(f)
    
    # 限制样本数量
    if sample_limit is not None:
        dataset = dataset[:sample_limit]
    
    # 初始化或加载结果文件
    if restart or not os.path.exists(detailed_output_path):
        evaluation_results = []
    else:
        with open(detailed_output_path, 'r') as f:
            evaluation_results = json.load(f)
    
    # 创建进度条
    total_samples = len(dataset)
    progress_bar = tqdm(total=total_samples, desc=f"评估 {dataset_config.name} 数据集")
    
    # 更新进度条到当前完成的数量
    progress_bar.update(len(evaluation_results))
    
    # 准备数据集
    selected_indices, has_hallucination_indices = prepare_dataset(dataset, dataset_config.name, sample_limit)
    
    # 使用线程池处理剩余样本
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, sample in enumerate(dataset[len(evaluation_results):]):
            args = (sample, idx, dataset_config, has_hallucination_indices, 
                   client, progress_bar, model, max_annotation_retries, 
                   api_retry_limit, prompt_method)
            future = executor.submit(evaluate_single_sample, args)
            futures.append(future)
        
        # 收集结果
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
                    
                    # 保存中间结果到详细结果文件
                    with results_lock:
                        with open(detailed_output_path, 'w') as f:
                            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
                    
                    # 更新进度条
                    with progress_lock:
                        progress_bar.update(1)
            except Exception as e:
                print(f"处理样本时发生错误: {str(e)}")
    
    progress_bar.close()
    
    # 打印评估总结
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
    """评估所有数据集"""
    all_results = {}
    
    # 创建共享的时间戳目录
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    safe_model_name = model.replace('/', '_')
    result_dir = f"results/{timestamp}_{safe_model_name}_{prompt_method}_{sample_limit}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 遍历所有数据集配置
    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        print(f"\n开始评估 {dataset_name.upper()} 数据集...")
        
        # 评估单个数据集并获取结果，传入共享的result_dir
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
            result_dir=result_dir  # 传入共享目录
        )
        
        # 保存结果到字典中
        all_results[dataset_name] = results
    
    # 保存汇总结果
    summary_path = os.path.join(result_dir, "evaluation_summary.csv")
    save_results_to_csv(all_results, model, prompt_method, summary_path)

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
                               'abab7-chat-preview',
                               'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
                               'glm-4v',
                               'local/MiniCPM-V-2_6',
                               'local/InternVL2-Llama3-76B'],
                      help='选择用的模型，默认为gpt-4o-2024-11-20')
    parser.add_argument('--prompt_method', type=str, default='vanilla',
                      choices=['vanilla', 'Criteria', '2-shot', 'Analyze-then-judge'],
                      help='选择使用的提示词方法，默认为vanilla')
    parser.add_argument('--max_workers', type=int, default=20,
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
        
        # 创建时间戳目录
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        safe_model_name = args.model.replace('/', '_')
        result_dir = f"results/{timestamp}_{safe_model_name}_{args.prompt_method}_{args.sample_limit}"
        
        print(f"\n开始评估 {args.dataset.upper()} 数据集...")
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