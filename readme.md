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
### Download the images for evaluation

Because the images are too large, we store them in google drive. You can download them from [here](https://drive.google.com/drive/folders/1C9Zdk4zZycJBRDU--Mo48Hk4TZG-Hn8S?usp=drive_link).
Then put the directory in the `evaluate` directory.

### Configure Environment Variables:
Create a `.env` file in the project root directory and add the following content:
```
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base_url
```

### Evaluating the Performance of Hallucination Detection

The main evaluation script is located in `evaluate/src/evaluate.py`. You can run the evaluation with the following command:

```bash
cd mhalo/evaluate
python src/evaluate.py   
```
### Explanation of the evaluation results

The evaluation results are stored in the `evaluate/results` directory with naming format: `evaluate/results/YYYY_MM_DD_HH_MM_SS_model-name_prompt-method_sample-limit`.

Each result directory contains:
- Five dataset-specific folders with detailed evaluation results
- An `evaluation_summary.csv` file with overall metrics

Example of evaluation summary:

| Metric | RLHF-V | M-HalDetect | Geo170K | MathV360K | MC | Average |
|--------|---------|-------------|----------|------------|-----|---------|
| Total Samples | 10 | 10 | 10 | 10 | 10 | 50 |
| Successful Samples | 10 | 10 | 9 | 9 | 9 | 47 |
| DSR | 1.0 | 1.0 | 0.9 | 0.9 | 0.9 | 0.94 |
| F1M | 0.250 | 0.0 | 0.301 | 0.287 | 0.549 | 0.278 |
| IOU_H | 0.156 | 0.0 | 0.091 | 0.203 | 0.483 | 0.187 |



### Available Parameters

- `--dataset`: The dataset to evaluate
  - Available options: ['all'] + all dataset names in DATASET_CONFIGS
  - Default: 'all'

- `--sample_limit`: The number of samples to evaluate
  - For example: `--sample_limit 50`
  - Default: 500

- `--model`: Select the model to use
  - Available options: ['gpt-4o-2024-11-20', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro-002', 'qwen-vl-max-0809', 'abab7-chat-preview', 'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo', 'glm-4v', 'local/MiniCPM-V-2_6', 'local/InternVL2-Llama3-76B']
  - Default: 'gpt-4o-2024-11-20'

- `--prompt_method`: Select the prompt method to use
  - Available options: ['vanilla', 'Criteria', '2-shot', 'Analyze-then-judge']
  - Default: 'vanilla'

- `--max_workers`: Maximum number of worker threads
  - Default: 20

- `--max_annotation_retries`: Maximum number of annotation retries
  - Default: 3

- `--api_retry_limit`: Maximum number of API call retries
  - Default: 3

### Example Commands

1. Evaluate all datasets:
```bash
python src/evaluate.py
```

2. Evaluate a specific dataset (e.g., mathv_360k) and limit the number of samples:
```bash
python src/evaluate.py --dataset mathv_360k --sample_limit 100
python src/evaluate.py --dataset rlhfv --sample_limit 10 
```

3. Use different models and prompt methods:
```bash
python src/evaluate.py --model qwen-vl-max-0809 --dataset MathV360K --prompt_method Analyze-then-judge
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

## Fine-tuning Data Set Description

### target_data.jsonl
This dataset is merged from the following three source datasets:

1. RLHF-Vision dataset (4,733 samples)
   - Source file: `ft_rlhfv_dataset_20250111_201136_new.json`

2. MHAL dataset (7,387 samples)
   - Source file: `ft_mhal_dataset_20250124_145127.json`

3. Math-Vision dataset (5,000 samples)
   - Source file: `ft_mathv_dataset_20250115_063847_n5000_p127_new.json`

To finetune your model,you should download the images from the google drive and put them in the `ft_data/images` directory using the [link](https://drive.google.com/drive/folders/19MB3Tf2YRJ0sNBFzLyuVhbNXvSfmYN5j?usp=drive_link)

Data processing instructions:
- Each data contains image path (image_path), prompt (prompt), dialog history (history) and reference answer (reference)
- The prompt contains a system message specially designed for hallucination detection and a user prompt template
- Images are stored in the images/rlhfv/、images/mhal/  and images/mathv/ directories according to the dataset type
