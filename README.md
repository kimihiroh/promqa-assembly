# ProMQA-Assembly: Multimodal Procedural QA Dataset on Assembly.
This repository contains code for "ProMQA-Assembly: Multimodal Procedural QA Dataset on Assembly." 

## Environment Setup

* OS: `Ubuntu 24.04.2 LTS x86_64`
* GPU: 4 A6000 (48GB)

### Virtual environment

API models 
```bash
conda create -y -n promqa-assembly python=3.12
conda activate promqa-assembly
pip install 
pip install pydot datasets
```

Deepseek-R1
```bash
conda create -y -n deepseek python=3.12
conda activate deepseek
pip install vllm
pip install pydot datasets
```

Qwen 2.5 VL
```bash
conda create -y -n qwen25vl python=3.12
conda activate qwen25vl
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord] torchvision pydot datasets
```

InternVL3
```bash
conda create -n internvl python=3.12
conda activate internvl
git clone https://github.com/OpenGVLab/InternVL.git
cd InternVL
pip install -r requirements.txt
pip install pydot datasets
```

LLaVA-Video
```bash
conda create llava-video python=3.10
conda activate llava-video
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install pydot datasets
```

## Dataset
* QA: https://huggingface.co/datasets/kimihiroh/promqa-assembly
* Instruction task graph: https://huggingface.co/datasets/kimihiroh/assembly101-graph
 
## Benchmarking

### Inference
Make sure to set an API key for each API model. Please check each `.sh` file for configurations.
* API models: `python src/benchmark/run_api.py --model_id <model_id>`
    * `<model_id>`: `gpt-4o-2024-11-20`
* DeepSeek-R1: `bash src/benchmark/run_deepseek_r1.py`
* Qwen2.5-VL: `bash src/benchmark/run_qwen.py`
* InternVL3: `bash src/benchmark/run_internvl.py`
* LLaVA-Video: `bash src/benchmark/run_llava_video.py`

### Evaluation
Make sure to set an API key, e.g., `export OPENAI_API_KEY=<your_key>`
```bash
bash src/evaluate/run.py --file <target_prediction_file>
```

## Citation (TBU)

If you find this work helpful in your research, please consider citing our work.
```bib
@misc{hasegawa-etal-2025-promqa-assembly,
      title={ProMQA-Assembly: Multimodal Procedural QA Dataset on Assembly},
      author={Hasegawa, Kimihiro and Imrattanatrai, Wiradee and Asada, Masaki and Holm, Susan and Wang, Yuran and Zhou, Vincent and Fukuda, Ken and Mitamura, Teruko},
      year={2025},
      url={https://github.com/kimihiroh/promqa-assembly},
}
```

## Issues/Questions

For any issues, questions, or requests, please create a [GitHub Issue](https://github.com/kimihiroh/promqa-assembly/issues). 