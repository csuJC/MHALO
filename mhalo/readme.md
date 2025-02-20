# MLLM Meta Evaluation

这是一个用于评测多模态大语言模型(MLLM)的项目。目前主要包含评测相关代码。
生成的带幻觉数据集在evaluate/data目录下,生成他们的代码在build/src下.

## 环境配置

1. 克隆仓库：
```bash
git clone https://github.com/YOUR_USERNAME/mllm_meta_eval.git
cd mllm_meta_eval
```

2. 创建并激活 conda 环境：
```bash
conda create -n mllm_eval python=3.9
conda activate mllm_eval
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境变量：
在项目根目录创建 `.env` 文件，添加以下内容：
```
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base_url
```

## 幻觉数据集构建部分

1.下载原数据集并放在项目的buld/data目录下，下载相应图片并放在build/images目录下

1.1 对于geo170k，需要从[Geo170K数据集](https://huggingface.co/datasets/Luckyjhg/Geo170K/tree/main)
下载qa_tuning.json文件，并放在build/data/Geo170K目录下,图片文件夹下载并解压后放在build/images目录下并改名为170k.

1.2 对于mathv360k，需要从[MathV360K数据集](https://huggingface.co/datasets/Zhiqiang007/MathV360K/tree/main)
下载train_samples_all_tuning.json，并放在build/data/mathv目录下,图片文件夹下载并解压后放在build/images目录下并改名为mathv

1.3 对于mhal，需要从[mhal-detect数据集](https://github.com/hendryx-scale/mhal-detect)
下载train_raw.json，并放在build/data/mhal-detect目录下,图片文件夹直接下载coco数据集并放在build/images目录下并改名为mhal.

1.4 对于[RLHF-V-Dataset](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset/tree/main):
下载图片，数据集直接从代码里加载


2.生成单个数据集可以分别运行
python build/src/process_mathv_360k.py,
python build/src/process_geo_170k.py,
python build/src/process_mhal_dataset.py,
python build/src/process_rlhfv_dataset.py,

后面可以添加生成样本数量的参数 只需要在代码后面加 --num_samples 1000,
例如 python build/src/process_geo_170k.py --num_samples 1000

3.运行build/src/build_dataset.py,生成带幻觉的数据集


## 评测使用方法

主要的评测脚本位于 `evaluate/src/evaluate.py`。你可以通过以下命令运行评测：

```bash
cd evaluate
python src/evaluate.py [参数]
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
