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

# 定义生成CoT并注入幻觉的prompt
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
def encode_image(image_path: str) -> str:
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_solution(client: OpenAI, question: str, answer: str, image_path: str) -> tuple[str, str, str, str]:
    """使用GPT-4-Vision生成带有思维链条的幻觉解答"""
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
        print(f"生成解答时出错: {str(e)}")
        return None, None, None, None

def extract_solutions(response: str) -> tuple[str, str, str]:
    """从GPT响应中提取原始解答和带标签/不带标签的幻觉解答"""
    # 提取原始解答
    pattern_original = r'<original_solution>(.*?)</original_solution>'
    match_original = re.search(pattern_original, response, re.DOTALL)
    
    if not match_original:
        print("没有找到原始解答")
        print(response)
        return None, None, None
    
    original_solution = match_original.group(1).strip()
    
    # 提取带标签的幻觉解答
    pattern_with_tag = r'<hallucinated_solution>(.*?)</hallucinated_solution>'
    match_with_tag = re.search(pattern_with_tag, response, re.DOTALL)
    
    if not match_with_tag:
        print("没有找到带标签的幻觉解答")
        print(response)
        return None, None, None
    
    solution_with_tag = match_with_tag.group(1).strip()
    
    if '<hallucination>' not in solution_with_tag:
        print("没有找到幻觉标签")
        print(response)
        return None, None, None
    
    # 生成不带标签的版本
    solution_without_tag = re.sub(r'<hallucination>(.*?)</hallucination>', r'\1', solution_with_tag)
    
    return original_solution, solution_with_tag, solution_without_tag

def generate_with_retry(client, question, answer, src_image):
    """带重试机制的生成函数"""
    max_retries = 3
    retry_count = 0
    
    print(f"\n开始处理数据:")
    print(f"问题: {question[:100]}...")
    print(f"答案: {answer[:100]}...")
    print(f"图片路径: {src_image}")
    
    while retry_count < max_retries:
        try:
            original_solution, hallucinated_solution, test_solution, api_response = generate_solution(
                client, question, answer, str(src_image)
            )
            
            if original_solution and hallucinated_solution and test_solution:
                print("✓ 解答生成成功!")
                return original_solution, hallucinated_solution, test_solution, api_response, True
            
            print(f"✗ 第 {retry_count + 1} 次生成失败，正在重试...")
            retry_count += 1
            
            if retry_count == max_retries:
                print(f"! 已达到最大重试次数 ({max_retries})，跳过该数据\n")
                return None, None, None, None, False
                
        except Exception as e:
            print(f"\n✗ 第 {retry_count + 1} 次生成失败:")
            print(f"错误信息: {str(e)}")
            retry_count += 1
            if retry_count == max_retries:
                return None, None, None, None, False
    
    return None, None, None, None, False

def process_single_item(item: Dict[str, Any], 
                       client: OpenAI,
                       input_images_dir: Path,
                       output_images_dir: Path,
                       lock: threading.Lock) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """处理单个数据项的函数
    
    Args:
        item: 要处理的数据项
        client: OpenAI客户端
        input_images_dir: 输入图片目录
        output_images_dir: 输出图片目录
        lock: 线程锁
        
    Returns:
        tuple: (处理后的数据, 失败信息)
    """
    try:
        if 'image' not in item:
            raise ValueError("缺少image字段")
        
        src_image = input_images_dir / item['image']
        if not src_image.exists():
            raise FileNotFoundError(f"图片文件不存在: {src_image}")
        
        question = item['conversations'][0]['value'] if item.get('conversations') else ''
        answer = item['conversations'][1]['value'] if len(item.get('conversations', [])) > 1 else ''
        
        original_solution, hallucinated_solution, test_solution, api_response, success = generate_with_retry(
            client, question, answer, str(src_image)
        )
        
        if not success:
            raise Exception("生成失败")
        
        # 使用线程锁保护文件复制操作
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

def process_dataset(num_samples: int = 10, num_threads: int = 4):
    """处理数据集并生成带有幻觉的版本
    
    Args:
        num_samples (int): 要生成的数据条数，默认为10
        num_threads (int): 使用的线程数，默认为4
    """
    # 设置OpenAI客户端
    client = OpenAI(
        base_url="https://one-api.glm.ai/v1",
        api_key = 'sk-ECLer7uvOSXsLd9fC7110b72A15f47848e418b2049C69361'
    )
    
    # 设置输入输出路径
    input_file = Path('build/data/mathv/train_samples_all_tuning.json')
    input_images_dir = Path('build/images/mathv')
    output_images_dir = Path('evaluate/images/mathv_360k')
    output_file = Path('evaluate/data/processed_mathv_360k.json')
    
    # 如果输出文件已存在，则删除它
    if output_file.exists():
        output_file.unlink()
    
    # 如果输出图片目录已存在，则删除它
    if output_images_dir.exists():
        shutil.rmtree(output_images_dir)
    
    # 创建输出目录
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取原始数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取总样本数
    total_samples = len(data)
    print(f"原始数据集共有 {total_samples} 条数据")
    print(f"目标生成 {num_samples} 条数据")
    print(f"使用 {num_threads} 个线程并行处理")
    
    # 初始化数据结构
    processed_data = []
    failed_samples = []
    lock = threading.Lock()
    
    # 创建进度条
    pbar = tqdm(total=num_samples, desc="处理MathV数据集")
    
    # 当前处理的数据索引
    current_index = 0
    batch_size = min(num_threads * 2, num_samples)  # 初始批次大小为线程数的2倍
    
    while len(processed_data) < num_samples and current_index < total_samples:
        # 计算这一批次需要处理的数量
        remaining_samples = num_samples - len(processed_data)
        current_batch_size = min(batch_size, remaining_samples)
        
        # 确保有足够的数据可以处理
        if current_index + current_batch_size > total_samples:
            current_batch_size = total_samples - current_index
            
        if current_batch_size <= 0:
            break
            
        current_batch = data[current_index:current_index + current_batch_size]
        
        print(f"\n当前处理进度：")
        print(f"已成功处理：{len(processed_data)}/{num_samples}")
        print(f"累计失败数：{len(failed_samples)}")
        print(f"当前批次大小：{len(current_batch)}")
        print(f"当前处理索引：{current_index}")
        
        # 使用线程池处理数据
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
                        print(f"\n处理样本时出错: {failed_info['error']}")
        
        # 更新当前索引，确保向前移动
        current_index += current_batch_size
        
        # 动态调整批次大小
        if successful_in_batch == 0:
            # 如果这个批次完全失败，减小批次大小以减少资源浪费
            batch_size = max(num_threads, batch_size // 2)
        elif successful_in_batch == current_batch_size:
            # 如果这个批次完全成功，适当增加批次大小
            batch_size = min(batch_size * 2, num_samples - len(processed_data))
        
        # 如果已经到达数据集末尾但还未达到目标数量
        if len(processed_data) < num_samples and current_index >= total_samples:
            print(f"\n警告：已到达数据集末尾，但仍未达到目标数量！")
            print(f"目标数量: {num_samples}")
            print(f"实际生成: {len(processed_data)}")
            break
    
    pbar.close()
    
    # 输出处理统计信息
    print(f"\n成功处理样本数: {len(processed_data)}")
    print(f"失败样本数: {len(failed_samples)}")
    print(f"总共尝试处理的数据条数: {current_index}")
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # 如果有失败的样本，保存失败记录
    if failed_samples:
        failed_samples_path = "evaluate/data/failed_mathv_360k_samples.json"
        with open(failed_samples_path, "w", encoding="utf-8") as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        print(f"失败样本信息已保存至: {failed_samples_path}")
    
    if len(processed_data) < num_samples:
        print(f"\n警告：未能达到目标数量！")
        print(f"目标数量: {num_samples}")
        print(f"实际生成: {len(processed_data)}")
        print(f"可能原因: 数据集中的样本不足或失败率过高")
    else:
        print(f"处理完成！")
        print(f"数据已保存至: {output_file}")
        print(f"图片已保存至: {output_images_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='处理数学数据集并生成带有幻觉的版本')
    parser.add_argument('--num_samples', type=int, default=500,
                      help='要生成的数据条数（默认：500）')
    parser.add_argument('--num_threads', type=int, default=8,
                      help='使用的线程数（默认：8）')
    
    args = parser.parse_args()
    process_dataset(args.num_samples, args.num_threads) 

