import json
import os
import base64
import re
from pathlib import Path
import random
from openai import OpenAI
from tqdm import tqdm
import shutil  # 添加这一行
import copy
from config import system_prompt_170k
import sys
#添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 定义 prompt
HALLUCINATION_PROMPT = system_prompt_170k

def encode_image(image_path: str) -> str:
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_solutions(response: str) -> tuple[str, str]:
    """从GPT响应中提取带标签和不带标签的幻觉解答"""
    pattern_with_tag = r'<hallucinated_solution>(.*?)</hallucinated_solution>'
    match_with_tag = re.search(pattern_with_tag, response, re.DOTALL)
    
    if not match_with_tag:
        return None, None
    
    solution_with_tag = match_with_tag.group(1).strip()
    
    # 检查是否包含<hallucination>标签
    if '<hallucination>' not in solution_with_tag:
        return None, None
    
    # 从带标签的解答中创建不带标签的版本
    solution_without_tag = re.sub(r'<hallucination>(.*?)</hallucination>', r'\1', solution_with_tag)
    
    return solution_with_tag, solution_without_tag

def generate_hallucinated_solution(client: OpenAI, question: str, original_solution: str, image_path: str) -> tuple[str, str, str]:
    """使用GPT-4-Vision生成带标签和不带标签的幻觉解答"""
    # 从环境变量获取API配置
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE')
    if not api_key or not api_base:
        raise ValueError("请设置环境变量 OPENAI_API_KEY 和 OPENAI_API_BASE")

    client = OpenAI(api_key=api_key, base_url=api_base)

    messages = [
        {
            "role": "system",
            "content": HALLUCINATION_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Question: {question}\n\nOriginal solution:\n{original_solution}"
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
        api_response_copy = copy.deepcopy(api_response)

        print("api_response", api_response )
        
        # 添加安全检查
        match = re.search(r'<hallucinated_solution>(.*?)</hallucinated_solution>', api_response, re.DOTALL)
        if not match:
            print("警告：API响应中未找到 hallucinated_solution 标签")
            return None, None, api_response_copy
            
        hallucinated_solution, solution_without_tag = extract_solutions(api_response)
        return hallucinated_solution, solution_without_tag, api_response_copy
    except Exception as e:
        print(f"生成幻觉解答时出错: {str(e)}")
        return None, None, None

def generate_with_retry(client, question, original_solution, src_image):
    """带重试机制的生成函数"""
    max_retries = 6  # 增加到6次重试
    retry_count = 0
    
    print(f"\n开始处理数据:")
    print(f"问题: {question[:100]}...")  # 只显示前100个字符
    print(f"原始解答: {original_solution[:100]}...")
    print(f"图片路径: {src_image}")
    
    while retry_count < max_retries:
        try:
            hallucinated_solution, test_solution, api_response = generate_hallucinated_solution(
                client, question, original_solution, str(src_image)
            )

            print("hallucinated_solution", hallucinated_solution)
            print("test_solution", test_solution)
            print("api_response_new", api_response)
            
            if hallucinated_solution and test_solution:
                print(f"✓ 生成成功!")
                return hallucinated_solution, test_solution, api_response, True
            
            print(f"✗ 第 {retry_count + 1} 次生成失败，正在重试...")
            retry_count += 1
            
            if retry_count == max_retries:
                print(f"! 已达到最大重试次数 ({max_retries})，跳过该数据\n")
                return None, None, None, False
                
        except Exception as e:
            print(f"\n✗ 第 {retry_count + 1} 次生成失败:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            
            retry_count += 1
            if retry_count == max_retries:
                print(f"! 已达到最大重试次数 ({max_retries})，跳过该数据\n")
                return None, None, None, False
    
    return None, None, None, False

def process_dataset(sample_size: int = 2000, overwrite: bool = False, batch_size: int = 4):
    """
    处理Geo170K数据集，生成带有视觉幻觉的解答
    
    Args:
        sample_size (int): 要处理的数据条数，默认为2000条
        overwrite (bool): 是否覆盖现有文件，默认为False
        batch_size (int): 并行处理的批次大小，默认为4
    """
    # 设置OpenAI客户端
    clients = [OpenAI(
        # api_key = 'sk-zwDE0esLqJIdNhf746D99c5e4b9f48889c836a01BcBb902f',
        # base_url="https://oneapi.lo-li.co/v1"
        base_url= "https://one-api.glm.ai/v1",
        api_key = 'sk-ECLer7uvOSXsLd9fC7110b72A15f47848e418b2049C69361'
    ) for _ in range(batch_size)]
    
    # 设置输出路径
    input_file = Path('/workspace/yishuo_sft/cys/mllm_meta_evl/build/data/Geo170K/qa_tuning.json')
    input_images_dir = Path('/workspace/yishuo_sft/cys/mllm_meta_evl/build/images/170k')
    output_images_dir = Path('/workspace/yishuo_sft/cys/mllm_meta_evl/evaluate/images/geo_170k')
    output_file = Path('/workspace/yishuo_sft/cys/mllm_meta_evl/evaluate/data/processed_geo_170k.json')
    
    # 如果输出图片目录已存在，则删除它
    if output_images_dir.exists():
        shutil.rmtree(output_images_dir)
    
    # 创建输出目录
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取原始数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 输出数据集信息
    total_samples = len(data)
    print(f"原始数据集共有 {total_samples} 条数据")
    print(f"目标生成 {sample_size} 条数据")
    print(f"使用 {batch_size} 个并行进程")

    # 选择前sample_size条数据
    data = data[:sample_size]
    
    # 处理每条数据
    processed_data = []
    failed_samples = []  # 用于记录失败的样本
    pbar = tqdm(total=sample_size, desc="处理Geo170K数据集")

    def process_batch(batch_items, client_idx):
        batch_results = []
        for item in batch_items:
            try:
                # 检查图片字段
                if 'image' not in item:
                    raise ValueError("缺少image字段")
                
                src_image = input_images_dir / item['image']
                if not src_image.exists():
                    raise FileNotFoundError(f"图片文件不存在: {src_image}")
                
                # 获取问题和原始解答
                question = item['conversations'][0]['value'] if item.get('conversations') else ''
                original_solution = item['conversations'][1]['value'] if len(item.get('conversations', [])) > 1 else ''
                
                # 使用GPT生成幻觉解答
                hallucinated_solution, test_solution, api_response, success = generate_with_retry(
                    clients[client_idx], question, original_solution, str(src_image)
                )
                
                if not success:
                    raise Exception("生成失败")
                
                # 复制图片到输出目录
                relative_image_path = item['image']  # 保持原始的相对路径
                dst_image = output_images_dir / relative_image_path
                dst_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_image, dst_image)
                
                # 构建新的数据结构
                batch_results.append({
                    'image': relative_image_path,
                    'question': question,
                    'original_solution': original_solution,
                    'hallucinated_solution': hallucinated_solution,
                    'test_solution': test_solution,
                    'api_response': api_response
                })
                pbar.update(1)
            except Exception as e:
                failed_samples.append({
                    "idx": len(processed_data) + len(failed_samples) + 1,
                    "question": question if 'question' in locals() else "",
                    "image_path": str(src_image) if 'src_image' in locals() else "",
                    "error": str(e)
                })
                print(f"\n处理样本时出错: {str(e)}")
        return batch_results

    # 将数据分成批次
    batch_data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for batch_idx, batch in enumerate(batch_data):
            client_idx = batch_idx % batch_size
            future = executor.submit(process_batch, batch, client_idx)
            futures.append(future)
        
        # 收集结果
        for future in as_completed(futures):
            batch_results = future.result()
            processed_data.extend(batch_results)

    pbar.close()

    # 输出处理统计信息
    print(f"\n成功处理样本数: {len(processed_data)}")
    print(f"失败样本数: {len(failed_samples)}")

    # 根据overwrite参数决定文件名
    if not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = output_file.with_name(f"{output_file.stem}_{timestamp}_{sample_size}{output_file.suffix}")

    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # 如果有失败的样本，保存失败记录
    if failed_samples:
        failed_samples_path = "evaluate/data/failed_geo_170k_samples.json"
        with open(failed_samples_path, "w", encoding="utf-8") as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        print(f"失败样本信息已保存至: {failed_samples_path}")

    print(f"处理完成！")
    print(f"数据已保存至: {output_file}")
    print(f"图片已保存至: {output_images_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='处理Geo170K数据集，生成带有视觉幻觉的解答')
    parser.add_argument('--num_samples', type=int, default=2000, help='要处理的数据条数（默认：2000）')
    parser.add_argument('--overwrite', action='store_true', help='是否覆盖现有文件（默认：否）')
    parser.add_argument('--batch_size', type=int, default=100, help='并行处理的批次大小（默认：4）')
    
    args = parser.parse_args()
    process_dataset(args.num_samples, args.overwrite, args.batch_size) 