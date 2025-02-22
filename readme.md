# MHALO: Evaluating MLLMs as Fine-grained Hallucination Detectors

This repo consists of core scripts for reproducing the main results of the paper **"MHALO: Evaluating MLLMs as Fine-grained Hallucination Detectors"**.

## Contributors

 Yishuo Cai $^1$ , Renjie Gu $^2$ 

$^1$ Peking University, $^2$ Central South University

If you have any questions or issues with the code, please send us an issue directly.

## Introduction

## Quick Start

Quick install the environment:

```
git clone https://github.com/csuJC/MHALO.git
cd MHALO 
conda create -n mhalo python=3.10
conda activate mhalo
pip install -r requirements.txt
```
### File Structure

The project is divided into three independent parts"

> `build` Folder - Corresponding to the process of generating hallucinated data
> 
> `evaluate` Folder - Corresponding to the evaluation of hallucination detection of different models
> 
> `ft_data` Folder - Corresponding to the data for fine-tuning

## Evaluating Hallucination Detection Performance


```
cd evaluate
```

### Configure Environment Variables:
Create a `.env` file in the project root directory and add the following content:
```
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base_url
```

## Building Hallucinated Data

1.Download the original dataset and put it in the build/data directory, download the corresponding images and put them in the build/images directory

1.1 For geo170k, you need to download the qa_tuning.json file from the [Geo170K dataset](https://huggingface.co/datasets/Luckyjhg/Geo170K/tree/main) and put it in the build/data/Geo170K directory, download the image folder and unzip it to the build/images directory and rename it to 170k.

1.2 For mathv360k, you need to download the train_samples_all_tuning.json file from the [MathV360K dataset](https://huggingface.co/datasets/Zhiqiang007/MathV360K/tree/main) and put it in the build/data/mathv directory, download the image folder and unzip it to the build/images directory and rename it to mathv

1.3 For mhal, you need to download the train_raw.json file from the [mhal-detect dataset](https://github.com/hendryx-scale/mhal-detect) and put it in the build/data/mhal-detect directory, download the image folder and unzip it to the build/images directory and rename it to mhal.

1.4 For [RLHF-V-Dataset](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset/tree/main):
Download the image, the dataset is loaded directly from the code


2.You can run the following scripts to generate single dataset:
```
python build/src/process_mathv_360k.py
```
```
python build/src/process_geo_170k.py
```
```
python build/src/process_mhal_dataset.py
```
```
python build/src/process_rlhfv_dataset.py
```

You can add the parameter to generate the number of samples, just add --num_samples 1000 after the code,
for example:

```
python build/src/process_geo_170k.py --num_samples 1000
```



## 评测使用方法

主要的评测脚本位于 `evaluate/src/evaluate.py`。你可以通过以下命令运行评测：

```bash
cd mhalo/evaluate
python src/evaluate.py  --model qwen-vl-max-0809 --sample_limit 10 --prompt_method Analyze-then-judge --api_retry_limit 1 --max_annotation_retries 1
```

### 可用参数

- `--dataset`: 选择要评测的数据集
  - 可选值: ['all', 'mathv_360k', 'geo_170k', 'mhal', 'rlhfv', 'multimath_300k']
  - 默认值: 'all'（评测所有数据集）

- `--sample_limit`: 评测样本数量限制（必须是10的倍数）
  - 例如: `--sample_limit 100`
  - 默认值: None（评测所有样本）

- `--restart`: 是否重新进行评测
  - 可选值: True/False
  - 默认值: True

- `--model`: 选择使用的模型
  - 可选值: ['gpt-4o', 'claude-3.5']
  - 默认值: 'gpt-4o'

- `--prompt_method`: 选择使用的提示词方法
  - 可选值: ['vanilla', 'cot', 'few-shot']
  - 默认值: 'vanilla'

- `--max_workers`: 最大工作线程数
  - 默认值: 8

### 示例命令

1. 评测所有数据集：
```bash
python src/evaluate.py
```

2. 评测特定数据集（如 mathv_360k）并限制样本数：
```bash
python src/evaluate.py --dataset mathv_360k --sample_limit 100
python src/evaluate.py --dataset rlhfv --sample_limit 10 --model tp_model
```

3. 使用不同的模型和提示词方法：
```bash
python src/evaluate.py --model claude-3.5 --prompt_method cot
```

## 评测结果

评测结果将保存在以下位置：
- 各数据集的详细结果：`evaluate/results/evaluation_results_{dataset_name}.json`
- 评测总结：`evaluate/results/evaluation_summary.csv`

## 微调数据集说明

### target_data.jsonl
该数据集由以下三个源数据集合并处理而成：

1. RLHF-Vision数据集 (4,733条数据)
   - 来源文件：`ft_rlhfv_dataset_20250111_201136_new.json`

2. MHAL数据集 (7,387条数据)
   - 来源文件：`ft_mhal_dataset_20250124_145127.json`

3. Math-Vision数据集 (5,000条数据)
   - 来源文件：`ft_mathv_dataset_20250115_063847_n5000_p127_new.json`

数据处理说明：
- 每条数据包含图片路径(image_path)、提示词(prompt)、对话历史(history)和参考答案(reference)
- 提示词包含了专门用于检测幻觉的系统消息和用户提示模板
- 图片按数据集类型分别存储在 images/rlhfv/、images/mhal/ 和 images/mathv/ 目录下
