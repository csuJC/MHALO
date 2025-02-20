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
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def download_coco_val2014_images(image_dir: str, image_ids: List[str]):
    """下载COCO 2014验证集的图片
    
    Args:
        image_dir: 图片保存目录
        image_ids: 需要下载的图片ID列表
    """
    os.makedirs(image_dir, exist_ok=True)
    
    # COCO 2014验证集的基础URL
    base_url = "http://images.cocodataset.org/val2014/"
    
    print(f"\n开始下载COCO验证集图片...")
    successful_downloads = 0
    failed_downloads = []
    
    for img_path in tqdm(image_ids, desc="下载图片"):
        if not img_path:  # 跳过空路径
            continue
            
        img_filename = os.path.basename(img_path)
        save_path = os.path.join(image_dir, img_filename)
        
        # 如果文件已存在，跳过下载
        if os.path.exists(save_path):
            successful_downloads += 1
            continue
            
        # 构建完整的URL
        url = base_url + img_filename
        
        try:
            # 使用curl下载图片，设置超时和重试
            cmd = f'curl -L --retry 3 --retry-delay 2 --connect-timeout 10 -o "{save_path}" "{url}"'
            result = subprocess.run(cmd, shell=True, capture_output=True)
            
            if result.returncode == 0 and os.path.exists(save_path):
                successful_downloads += 1
            else:
                failed_downloads.append(img_path)
                if os.path.exists(save_path):
                    os.remove(save_path)  # 删除可能的不完整文件
                    
        except Exception as e:
            print(f"下载图片 {img_filename} 时出错: {str(e)}")
            failed_downloads.append(img_path)
            if os.path.exists(save_path):
                os.remove(save_path)
    
    print(f"\n图片下载完成:")
    print(f"成功下载: {successful_downloads} 张图片")
    print(f"下载失败: {len(failed_downloads)} 张图片")
    if failed_downloads:
        print("\n以下图片下载失败:")
        for img in failed_downloads[:10]:  # 只显示前10个
            print(f"- {img}")
        if len(failed_downloads) > 10:
            print(f"... 还有 {len(failed_downloads) - 10} 张图片未显示")

def extract_hallucination_spans_mhal(original: str, corrected: str) -> List[Dict[str, int]]:
    """提取mhal-detect数据集中幻觉文本的起始和结束位置"""
    # 获取差异ID
    original_diff_ids, _ = get_diff_ids(original, corrected)
    
    if not original_diff_ids:
        return []
    
    # 将文本分割成单词，用于构建spans
    original_words = split_into_words(original)
    
    # 按顺序处理差异ID，合并连续的差异
    spans = []
    start = original_diff_ids[0]
    current_end = start + 1
    
    for i in range(1, len(original_diff_ids)):
        current_id = original_diff_ids[i]
        # 只有当差异ID完全连续时才合并
        if current_id == current_end:
            current_end += 1
        else:
            # 添加当前span
            if current_end > start:
                spans.append({
                    "start": start,
                    "end": current_end,
                    "text": " ".join(original_words[start:current_end])
                })
            # 开始新的span
            start = current_id
            current_end = start + 1
    
    # 添加最后一个span
    if current_end > start and start < len(original_words):
        spans.append({
            "start": start,
            "end": current_end,
            "text": " ".join(original_words[start:current_end])
        })
    
    return spans

def add_hallucination_tags_mhal(text: str, spans: List[Dict[str, int]]) -> str:
    """在mhal-detect数据集的文本中添加幻觉标记"""
    if not spans:
        return text
        
    words = split_into_words(text)
    tagged_text = ""
    last_end = 0
    
    # 合并重叠或紧邻的spans
    merged_spans = []
    sorted_spans = sorted(spans, key=lambda x: x["start"])
    current_span = sorted_spans[0]
    
    for next_span in sorted_spans[1:]:
        # 只有当两个span完全相邻或重叠时才合并
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
    
    # 使用合并后的spans添加标记
    for span in merged_spans:
        # 添加span前的正常文本
        if span["start"] > last_end:
            tagged_text += " ".join(words[last_end:span["start"]]) + " "
        
        # 添加带标记的文本
        hallucinated_text = " ".join(words[span["start"]:span["end"]])
        if hallucinated_text.strip():  # 只有当文本非空时才添加标记
            tagged_text += f"<hallucination>{hallucinated_text}</hallucination> "
        
        last_end = span["end"]
    
    # 添加最后剩余的正常文本
    if last_end < len(words):
        tagged_text += " ".join(words[last_end:])
    
    return tagged_text.strip()

def process_mhal_sample(sample: Dict, index: int) -> Dict:
    """处理mhal-detect数据集的单个数据样本"""
    original = sample["response"]
    annotations = sample["annotations"]
    prompt = sample["question"]
    
    # 提取幻觉span（只使用INACCURATE标签）
    char_spans = []
    for ann in annotations:
        if ann["label"] == "INACCURATE":
            # 确保span的文本与原始文本匹配
            span_text = original[ann["start"]:ann["end"]]
            if span_text == ann["text"]:
                char_spans.append({
                    "start": ann["start"],
                    "end": ann["end"],
                    "text": span_text
                })
            else:
                print(f"警告: span文本不匹配:\n预期: {ann['text']}\n实际: {span_text}")
    
    # 按起始位置排序
    char_spans.sort(key=lambda x: x["start"])
    
    # 合并重叠的spans
    merged_char_spans = []
    if char_spans:
        current_span = char_spans[0]
        for next_span in char_spans[1:]:
            if next_span["start"] <= current_span["end"]:
                # 合并重叠的spans
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
    
    # 添加标记
    result = original
    offset = 0
    for span in merged_char_spans:
        start = span["start"] + offset
        end = span["end"] + offset
        hallucination_text = result[start:end]
        marked_text = f"<hallucination>{hallucination_text}</hallucination>"
        result = result[:start] + marked_text + result[end:]
        offset += len(marked_text) - len(hallucination_text)
    
    # 构造图片路径
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
    """处理mhal-detect数据集
    
    Args:
        num_samples (int): 要处理的数据条数，默认为500。
                          其中90%为有幻觉样本，10%为无幻觉样本。
    """
    print("开始加载mhal-detect数据集...")
    
    # 计算需要的有幻觉和无幻觉样本数量
    required_hallucination_samples = int(num_samples * 0.9)
    required_clean_samples = num_samples - required_hallucination_samples
    
    print(f"目标总样本数: {num_samples}")
    print(f"目标有幻觉样本数: {required_hallucination_samples}")
    print(f"目标无幻觉样本数: {required_clean_samples}")
    
    # 使用绝对路径加载数据集
    dataset_path = os.path.join(build_dir, 'data', 'mhal-detect', 'val_raw.json')
    print(f"尝试从以下路径加载数据集: {dataset_path}")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        return
    
    print(f"原始数据集共有 {len(dataset)} 条数据")
    
    # 筛选包含图片的数据
    dataset_with_images = [sample for sample in dataset if sample.get('image')]
    print(f"其中包含图片的数据有 {len(dataset_with_images)} 条")
    
    if len(dataset_with_images) < num_samples:
        print(f"警告：包含图片的数据数量({len(dataset_with_images)})少于请求的样本数量({num_samples})")
        dataset = dataset_with_images
    else:
        # 随机选择指定数量的包含图片的数据
        random.seed(42)
        dataset = random.sample(dataset_with_images, num_samples)
    
    # 处理数据
    processed_data = []
    for index, sample in enumerate(tqdm(dataset, desc="处理数据")):
        processed_sample = process_mhal_sample(sample, index)
        # 对于无幻觉样本，设置original_solution
        if not processed_sample['hallucination_spans']:
            processed_sample['original_solution'] = processed_sample['test_solution']
        processed_data.append(processed_sample)
    
    # 分离有幻觉和无幻觉的样本
    samples_with_hallucination = [s for s in processed_data if s['hallucination_spans']]
    samples_without_hallucination = [s for s in processed_data if not s['hallucination_spans']]
    
    print(f"\n初步处理完成，发现:")
    print(f"有幻觉样本: {len(samples_with_hallucination)} 条")
    print(f"无幻觉样本: {len(samples_without_hallucination)} 条")
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 根据比例选择样本
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
    
    # 合并最终数据
    final_processed_data = final_hallucination_samples + final_clean_samples
    random.shuffle(final_processed_data)  # 随机打乱顺序
    
    print(f"\n最终数据统计:")
    print(f"有幻觉样本: {len(final_hallucination_samples)} 条")
    print(f"无幻觉样本: {len(final_clean_samples)} 条")
    print(f"总样本数: {len(final_processed_data)} 条")
    
    # 设置本地图片目录和输出目录
    local_image_dir = os.path.join(build_dir, 'images', 'mhal')
    image_output_dir = os.path.join(build_dir, '..', 'evaluate', 'images', 'mhal')
    os.makedirs(image_output_dir, exist_ok=True)
    
    # 清空图片输出目录
    for filename in os.listdir(image_output_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(image_output_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"删除文件 {file_path} 时出错: {str(e)}")
    
    # 从本地目录复制图片到输出目录
    print("\n开始从本地目录复制图片...")
    successful_copies = 0
    failed_copies = []
    
    for sample in tqdm(final_processed_data, desc="复制图片"):
        if not sample.get('image_path'):
            continue
            
        img_filename = os.path.basename(sample['image_path'])
        src_path = os.path.join(local_image_dir, img_filename)
        dst_path = os.path.join(image_output_dir, img_filename)
        
        if not os.path.exists(src_path):
            print(f"警告：本地图片不存在: {src_path}")
            failed_copies.append(img_filename)
            continue
            
        try:
            import shutil
            shutil.copy2(src_path, dst_path)
            successful_copies += 1
        except Exception as e:
            print(f"复制图片 {img_filename} 时出错: {str(e)}")
            failed_copies.append(img_filename)
    
    print(f"\n图片复制完成:")
    print(f"成功复制: {successful_copies} 张图片")
    print(f"复制失败: {len(failed_copies)} 张图片")
    if failed_copies:
        print("\n以下图片复制失败:")
        for img in failed_copies[:10]:
            print(f"- {img}")
        if len(failed_copies) > 10:
            print(f"... 还有 {len(failed_copies) - 10} 张图片未显示")
    
    # 修改输出路径
    output_path = os.path.join(build_dir, '..', 'evaluate', 'data', 'processed_mhal_dataset_new.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！")
    print(f"数据已保存至: {output_path}")
    print(f"图片已保存至: {image_output_dir}")

def test_hallucination_tagging():
    """测试幻觉标记的添加逻辑"""
    test_cases = [
        {
            "name": "基础测试 - 使用原始和修正文本",
            "original": "The couch is situated in the center of the room.",
            "corrected": "The couch is in the room.",
            "expected_spans": [
                {"start": 3, "end": 4, "text": "situated"},
                {"start": 5, "end": 8, "text": "in the center"}
            ]
        },
        {
            "name": "连续幻觉测试 - 使用原始和修正文本",
            "original": "The image shows a living room with a large window. The walls are painted blue.",
            "corrected": "The image contains a window. The walls are white.",
            "expected_spans": [
                {"start": 2, "end": 7, "text": "shows a living room with"},
                {"start": 8, "end": 9, "text": "large"},
                {"start": 13, "end": 15, "text": "painted blue"}
            ]
        },
        {
            "name": "直接使用spans测试",
            "text": "The kitchen features modern appliances and granite countertops. The cabinets are wooden.",
            "spans": [
                {"start": 2, "end": 5, "text": "kitchen features modern"},
                {"start": 8, "end": 10, "text": "granite countertops"}
            ]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n测试用例: {test_case['name']}")
        
        if "original" in test_case and "corrected" in test_case:
            # 使用extract_hallucination_spans_mhal提取spans
            print("原始文本:", test_case["original"])
            print("修正文本:", test_case["corrected"])
            spans = extract_hallucination_spans_mhal(test_case["original"], test_case["corrected"])
            text = test_case["original"]
        else:
            # 直接使用提供的spans
            print("文本:", test_case["text"])
            spans = test_case["spans"]
            text = test_case["text"]
            
        print("幻觉spans:", spans)
        
        # 使用标记方法
        tagged = add_hallucination_tags_mhal(text, spans)
        print("\n标记结果:", tagged)
        
        # 验证结果
        words = split_into_words(text)
        for span in spans:
            hallucinated_text = " ".join(words[span["start"]:span["end"]])
            if hallucinated_text.strip() and f"<hallucination>{hallucinated_text}</hallucination>" not in tagged:
                print(f"警告: 未找到预期的标记文本: {hallucinated_text}")
        
        if "expected_spans" in test_case:
            # 验证提取的spans是否符合预期
            for expected_span in test_case["expected_spans"]:
                found = False
                for span in spans:
                    if span["start"] == expected_span["start"] and span["end"] == expected_span["end"]:
                        found = True
                        break
                if not found:
                    print(f"警告: 未找到预期的span: {expected_span}")

def test_specific_case():
    """测试特定样本的处理逻辑"""
    # 测试样本
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
    
    # 处理样本
    processed = process_mhal_sample(test_sample, 0)
    
    # 打印处理结果
    print("\n原始文本:")
    print(test_sample["response"])
    print("\n标注信息:")
    for ann in test_sample["annotations"]:
        print(f"Label: {ann['label']}")
        print(f"Text: {ann['text']}")
        print(f"Range: {ann['start']}-{ann['end']}")
    
    print("\n处理后的文本:")
    print(processed["hallucinated_solution"])
    print("\n幻觉spans:")
    for span in processed["hallucination_spans"]:
        print(f"Start: {span['start']}, End: {span['end']}, Text: {span['text']}")
    
    # 验证处理结果
    # 1. 检查是否只处理了INACCURATE标签
    inaccurate_texts = [ann["text"] for ann in test_sample["annotations"] if ann["label"] == "INACCURATE"]
    print("\n验证结果:")
    print("1. INACCURATE标签文本:")
    for text in inaccurate_texts:
        print(f"- {text}")
        # 检查是否都被正确标记
        if f"<hallucination>{text}</hallucination>" not in processed["hallucinated_solution"]:
            print(f"警告: 未找到预期的标记文本: {text}")
    
    # 2. 检查其他标签的文本是否被错误标记
    other_texts = [ann["text"] for ann in test_sample["annotations"] if ann["label"] != "INACCURATE"]
    print("\n2. 非INACCURATE标签文本:")
    for text in other_texts:
        print(f"- {text}")
        if f"<hallucination>{text}</hallucination>" in processed["hallucinated_solution"]:
            print(f"警告: 发现不应该被标记的文本: {text}")
    
    # 3. 检查spans的连续性
    print("\n3. 检查spans的连续性:")
    for i in range(len(processed["hallucination_spans"]) - 1):
        current_span = processed["hallucination_spans"][i]
        next_span = processed["hallucination_spans"][i + 1]
        if current_span["end"] > next_span["start"]:
            print(f"警告: 发现重叠的spans: {current_span} 和 {next_span}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='处理mhal-detect数据集')
    parser.add_argument('--num_samples', type=int, default=500,
                      help='要处理的数据条数（默认：500）')
    parser.add_argument('--test', action='store_true',
                      help='运行标记测试')
    parser.add_argument('--test_specific', action='store_true',
                      help='运行特定样本测试')
    
    args = parser.parse_args()
    
    if args.test:
        test_hallucination_tagging()
    elif args.test_specific:
        test_specific_case()
    else:
        process_dataset(args.num_samples)