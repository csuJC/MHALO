import json
import sys
import os
from typing import List, Dict
import difflib
from datasets import load_dataset
import random
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加build目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/build/src
build_dir = os.path.dirname(current_dir)  # /path/to/build
sys.path.insert(0, build_dir)

from utils.diff_lib import get_diff_ids, split_into_words

def extract_hallucination_spans(rejected: str, chosen: str) -> List[Dict[str, int]]:
    """提取rejected文本中的幻觉片段的span"""
    # 将文本分割成单词
    rejected_words = split_into_words(rejected)
    chosen_words = split_into_words(chosen)
    
    # 使用difflib来获取差异
    matcher = difflib.SequenceMatcher(None, rejected_words, chosen_words)
    rejected_diff_indices = set(range(len(rejected_words)))
    
    # 移除匹配的部分
    for match in matcher.get_matching_blocks():
        for i in range(match.size):
            if match.a + i in rejected_diff_indices:
                rejected_diff_indices.remove(match.a + i)
    
    rejected_diff_ids = sorted(list(rejected_diff_indices))
    
    # 将连续的索引组合成span
    spans = []
    if not rejected_diff_ids:
        return spans
    
    # 初始化第一个span
    current_span_start = rejected_diff_ids[0]
    current_words = []
    
    # 遍历所有差异索引
    for i, idx in enumerate(rejected_diff_ids):
        current_words.append(rejected_words[idx])
        
        # 如果是最后一个索引或者下一个索引不连续
        if i == len(rejected_diff_ids) - 1 or rejected_diff_ids[i + 1] != idx + 1:
            # 保存当前span
            spans.append({
                "start": current_span_start,
                "end": idx + 1,
                "text": " ".join(current_words)
            })
            # 如果不是最后一个索引，开始新的span
            if i < len(rejected_diff_ids) - 1:
                current_span_start = rejected_diff_ids[i + 1]
                current_words = []
    
    return spans

def add_hallucination_tags(text: str, spans: List[Dict[str, int]]) -> str:
    """在文本中添加幻觉标记"""
    words = split_into_words(text)
    tagged_words = words.copy()
    
    # 从后向前添加标记，避免索引变化
    for span in reversed(spans):
        start, end = span["start"], span["end"]
        tagged_words[start:end] = [f"<hallucination>{' '.join(words[start:end])}</hallucination>"]
    
    return " ".join(tagged_words)

def process_sample(sample: Dict) -> Dict:
    """处理单个数据样本"""
    text_data = json.loads(sample["text"])
    rejected = text_data["rejected"]
    chosen = text_data["chosen"]
    prompt = text_data.get("question", "What do you observe in this image?")  # 添加默认prompt
    
    # 提取幻觉span
    hallucination_spans = extract_hallucination_spans(rejected, chosen)
    
    # 添加标记
    tagged_text = add_hallucination_tags(rejected, hallucination_spans)
    
    

    return {
        "id": sample.get("idx", "unknown"),  # 从原始数据集的idx字段获取id
        "image_path": sample["image_path"],
        "original_solution": chosen,
        "hallucinated_solution": tagged_text,
        "test_solution": rejected,
        "prompt": prompt  # 添加prompt字段
    }

def process_dataset(num_samples: int = 500):
    """处理RLHF-V数据集
    
    Args:
        num_samples (int): 要生成的数据条数，默认为500
    """
    print("开始加载RLHF-V数据集...")
    dataset = load_dataset("HaoyeZhang/RLHF-V-Dataset")
    train_data = list(dataset['train'])

    # 只取前num_samples条数据
    train_data = train_data[:num_samples]
    
    # 确保图片输出目录存在
    image_output_dir = "evaluate/images/rlhfv"
    os.makedirs(image_output_dir, exist_ok=True)
    
    # 清空图片输出目录
    for filename in os.listdir(image_output_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(image_output_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"删除文件 {file_path} 时出错: {str(e)}")
    
    processed_data = []
    failed_samples = []
    pbar = tqdm(total=num_samples, desc="处理RLHF-V数据集")
    
    for sample in train_data:
        try:
            processed_sample = process_sample(sample)
            
            # 处理图片
            if sample.get('image_path'):
                src_image_path = os.path.join('build/images/rlhfv', sample['image_path'])
                dst_image_path = os.path.join(image_output_dir, os.path.basename(sample['image_path']))
                
                if os.path.exists(src_image_path):
                    os.system(f'cp "{src_image_path}" "{dst_image_path}"')
                    # 只保存文件名
                    processed_sample['image_path'] = os.path.basename(sample['image_path'])
                else:
                    raise FileNotFoundError(f"图片文件不存在: {src_image_path}")
            
            processed_data.append(processed_sample)
            pbar.update(1)
            
        except Exception as e:
            failed_samples.append({
                "sample_id": sample.get("idx", "unknown"),
                "error": str(e)
            })
            print(f"\n处理样本 {sample.get('idx', 'unknown')} 时出错: {str(e)}")
            continue
    
    pbar.close()
    
    # 输出处理统计信息
    print(f"\n成功处理样本数: {len(processed_data)}")
    print(f"失败样本数: {len(failed_samples)}")
    
    # 修改输出路径
    output_path = "evaluate/data/processed_rlhfv_dataset.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存处理后的数据
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # 如果有失败的样本，保存失败记录
    if failed_samples:
        failed_samples_path = "evaluate/data/failed_rlhfv_samples.json"
        with open(failed_samples_path, "w", encoding="utf-8") as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        print(f"失败样本信息已保存至: {failed_samples_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='处理RLHF-V数据集')
    parser.add_argument('--num_samples', type=int, default=2000,
                      help='要生成的数据条数（默认2000）')
    
    args = parser.parse_args()
    process_dataset(args.num_samples) 