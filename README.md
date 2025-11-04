# ProMQA-Assembly: Multimodal Procedural QA Dataset on Assembly.
This is the official repository for "[ProMQA-Assembly: Multimodal Procedural QA Dataset on Assembly](https://arxiv.org/abs/2509.02949)" (Hasegawa et al., arXiv 2025).

It contains code and data for:
* Data annotation: preprocess, generation, and verification
* Benchmarking: download, prediction, and evaluation

## News
* 2025/11/04: Additional 255 QA pairs are added. In total, 646 QAs are now available (`data/all_v1.json`).
* 2025/09/03: 391 QA pairs are now available (`data/all_v0.json`).

## Overview
ProMQA-Assembly is an evaluation QA dataset for multimodal procedural activity understanding on assembly.

![Overview](https://github.com/kimihiroh/promqa-assembly/blob/main/docs/overview.png)


## Environment Setup

* OS: `Ubuntu 24.04.2 LTS x86_64`
* GPU: 4 A6000 (48GB)

### Virtual environment

```bash
conda create -y -n promqa-assembly python=3.12
conda activate promqa-assembly
pip install tqdm pre-commit anthropic openai google-genai pydot datasets pillow PyDrive2 networkx litellm
pre-commit install
```

### Others
* Install `nvm` if not available

## Data

### Download Assembly 101

#### Preprocessed Data
You can download the pre-sampled frames (360p) of Assembly101 from [our HuggingFace Dataset repo](https://huggingface.co/datasets/kimihiroh/promqa-assembly-frames):
```bash
cd <dirpath_hf>
git clone https://huggingface.co/datasets/kimihiroh/promqa-assembly-frames
cd <...>/promqa-assembly
bash src/unzip_all.sh <dirpath_hf>/promqa-assembly-frames
```
If you want the data as video in original resolution, please check [the following instruction](####full-data).

#### Full Data
* Preparation
    * Submit access request to [Assembly 101 Google Drive](https://drive.google.com/drive/folders/1nh8PHwEw04zxkkkKlfm4fsR3IPEDvLKj)
    * Clone https://github.com/assembly-101/assembly101-download-scripts in `repos`
* Authentification [[ref](https://github.com/assembly-101/assembly101-download-scripts)]
    * Follow [the authentification process](https://docs.iterative.ai/PyDrive2/quickstart/#authentication), and obtain `client_secrets.json`. [[ref](https://github.com/assembly-101/assembly101-download-scripts/issues/12#issuecomment-1518940304)]
    * Update `client_config` in `settings.yaml`.
    * Download `authenticate.py` and `settings.yaml` on your local machine and run `authenticate.py` with an accessible browser to generate `credentials.json`. [[ref](https://github.com/assembly-101/assembly101-download-scripts?tab=readme-ov-file#authentication)]
    * Upload `credentials.json` to this directory.
* Download recording
    * Modify `download.py`
    ```python
    -    os.rename(f'{file_name}', f'{out_path}/{file_name}')
    +    shutil.copy(f'{file_name}', f'{out_path}/{file_name}')
    +    os.remove(f'{file_name}')
    ```
    * Run the following command
    ```bash
    python download.py --videos all --views all --outDir <output_dir>
    ```
* Download action labels
    * Clone https://github.com/assembly-101/assembly101-annotations in `repos`
    * Manually download files from [here](https://drive.google.com/drive/folders/1QoT-hIiKUrSHMxYBKHvWpW9Z9aCznJB7?usp=sharing) and places files in corresponding folders [[ref](https://github.com/assembly-101/assembly101-annotations)]
* Download mistake labels
    * Clone https://github.com/assembly-101/assembly101-mistake-detection in `repos`
* Note
    * The download operation may need to be restarted after waiting for a day. [[ref](https://research.google.com/colaboratory/faq.html#drive-quota)]
Now, you can skip to [the benchmark section](##benchmarking) if you do not annotate QAs by yourself.


## QA&Graph Annotation

### Preprocess
```bash
# collect data and format into json with annotation updates
bash src/preprocess/format_annotation.sh <output_dir>
bash src/preprocess/create_example.sh
bash src/preprocess/sample.sh 800
```

### Graph Annotation
```bash
# start interface backend in e.g., tmux
bash src/graph-annotation/backend/start.sh .data/preprocess/graph/ .data/graph/
# start interface frontend in e.g., tmux
cd ../frontend/
npm install
ln -s <output_dir> ./public/
ln -s ./data/parts/original ./public/
npm start

# postprocess
bash src/graph-annotation/create_image.sh
```

### QA Generation
```bash
# generation
bash src/qa-generation/run.sh samples_10.json
# postprocess
bash src/qa-generation/postprocess.sh \
    samples_10.default.gpt-4o-2024-11-20.coarse+fine_all.False.json
# create corresponding task graphs w/ status
bash src/graph-annotation/create_image_w_status.sh samples_10.json
```

### QA Verification
```bash
# start interface backend in e.g., tmux
bash src/qa-verification/backend/start.sh ./data/qa-generation/ ./data/qa-verification/
# start interface frontend in e.g., tmux
cd src/qa-verification/frontend/
npm install
ln -s <output_dir> ./public/
ln -s ./data/parts/original ./public/
ln -s ./data/graph/raw/ ./public/
ln -s ./data/graph/status/ ./public/
```

## Benchmarking

### Preprocess
Sample frames from videos. You can skip this if you download the pre-sampled frames.
```bash
bash src/benchmark/sample_frame.sh <output_dir> <output_dir_frame>
```

### Inference
- [ ] remove
Make sure to set an API key as an environment variable, e.g., `export OPENAI_API_KEY=<your_key>`
```bash
bash src/benchmark/run_api.sh <output_dir_frame> gpt-4o-2024-11-20 default
bash src/benchmark/run_api.sh <output_dir_frame> gpt-4o-2024-11-20 cot
bash src/benchmark/run_api.sh <output_dir_frame> gpt-4o-2024-11-20 text

bash src/benchmark/run_api.sh <output_dir_frame> gpt-5-2025-08-07 reasoning minimal
bash src/benchmark/run_api.sh <output_dir_frame> gpt-5-2025-08-07 reasoning medium
bash src/benchmark/run_api.sh <output_dir_frame> gpt-5-2025-08-07 text minimal

bash src/benchmark/run_api.sh <output_dir_frame> claude-3-7-sonnet-20250219 default
bash src/benchmark/run_api.sh <output_dir_frame> claude-3-7-sonnet-20250219 cot
bash src/benchmark/run_api.sh <output_dir_frame> claude-3-7-sonnet-20250219 text
bash src/benchmark/run_api.sh <output_dir_frame> claude-3-7-sonnet-20250219 reasoning

bash src/benchmark/run_api.sh <output_dir_frame> gemini-2.5-pro default
bash src/benchmark/run_api.sh <output_dir_frame> gemini-2.5-pro cot
bash src/benchmark/run_api.sh <output_dir_frame> gemini-2.5-pro text
```

### Evaluation
Make sure to set an API key, e.g., `export OPENAI_API_KEY=<your_key>`
```bash
bash src/benchmark/evaluate.sh <target_prediction_filename>
```

## Citation

If you find this work helpful in your research, please consider citing our work.
```bib
@article{hasegawa-etal-2025-promqa-assembly,
      title={ProMQA-Assembly: Multimodal Procedural QA Dataset on Assembly},
      author={Hasegawa, Kimihiro and Imrattanatrai, Wiradee and Asada, Masaki and Holm, Susan and Wang, Yuran and Zhou, Vincent and Fukuda, Ken and Mitamura, Teruko},
      year={2025},
      archivePrefix={arXiv},
      eprint={2509.02949},
}
```

## Issues/Questions

For any issues, questions, or requests, please create a [GitHub Issue](https://github.com/kimihiroh/promqa-assembly/issues).
