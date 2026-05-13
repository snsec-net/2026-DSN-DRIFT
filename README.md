# DRIFT: Drift-Resilient Invariant-Feature Transformer for DGA Detection

This repository contains the official implementation of the paper **"DRIFT: Drift-Resilient Invariant-Feature Transformer for DGA Detection"**. 

Authors: Chaeyoung Lee*, Chaeri Jung*, Seonghoon Jeong
Advisor: Prof. Seonghoon Jeong
(* Equal Contribution)
📄 Paper (DSN 2026): Coming soon
📄 arXiv: https://arxiv.org/abs/2605.10436

This study proposes **DRIFT**, a drift-resilient DGA detection framework that mitigates temporal performance degradation by leveraging a hybrid tokenization strategy (character-level and subword-level encodings) and multi-task self-supervised pre-training.

## Environment Setup

This project is implemented using the **PyTorch** framework and leverages the **Hugging Face `transformers`** and **`tokenizers`** libraries.

**Installation:**
```bash
# Clone the repository
git clone https://github.com/snsec-net/2026-DSN-DRIFT.git
cd 2026-DSN-DRIFT

# Create and activate a virtual environment
conda create -n drift
conda activate drift

# Install dependencies
conda install -y python=3.14 pip cuda-nvcc=13.0
pip install pandas pyarrow numpy scipy matplotlib ipython scikit-learn pillow jupyter openpyxl tqdm seaborn wandb transformers polars tokenizers torchinfo "torch>=2.10.0" torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
```

### Alternative Setup (Troubleshooting)
If you encounter any dependency issues with the standard `pip` installation above, you can try creating the environment and installing all packages directly via `conda` in a single command:

```bash
ENV_NAME="drift"
conda create -n $ENV_NAME python=3.13 pip pandas pyarrow numpy scipy matplotlib ipython scikit-learn pillow jupyter openpyxl tqdm seaborn wandb transformers polars tokenizers pyarrow torchinfo pytorch=2.10 torchvision torchaudio

conda activate $ENV_NAME
```

## Dataset Preparation

The dataset used in this study is available on **IEEE Dataport**. The provided data is **already fully preprocessed** (including deterministic Effective SLD extraction) and is ready to be used directly as input for the model.

* **Dataset Link:** https://ieee-dataport.org/documents/longitudinal-benign-and-dga-domain-name-dataset
* **Contents:** The longitudinal dataset covers a nine-year period (2017–2025), including preprocessed benign domains (from Alexa/Tranco) and DGA domains (from DGArchive).

**Important Directory Structure:**
To ensure the provided scripts run correctly, the downloaded data must be organized as follows. Specifically, the `DRIFT_input_eSLD` folder must be located directly inside the `dataset` directory:

```text
DRIFT/
├── dataset/
│   └── DRIFT_input_eSLD/        <-- Ready-to-use preprocessed data
│       ├── T17_benign_train.parquet
│       ├── T18_benign_train.parquet
│       ├── T19_benign_train.parquet
│       └── ...
├── pretrain.py
├── finetuning.py
└── ...
```

> **Note:** Since the data is already preprocessed, you do not need to run any additional extraction or normalization scripts. For detailed information on the preprocessing rules applied to create this dataset, please refer to Section III.1 of the paper.

## How to Run

Model training is divided into two main phases: 
  1) Self-supervised Pre-training
  2) Supervised Fine-tuning for DGA detection

### 1. Self-supervised Pre-training
The subword and character backbones are pre-trained independently using three auxiliary tasks: Masked Token Prediction (MTP), Token Position Prediction (TPP), and Token Order Verification (TOV).

```bash
# Pre-train the Subword-based Transformer Backbone
python pretrain.py --mode subword --save subword.pt --use_bf16 --no_wandb

# Pre-train the Character-based Transformer Backbone
python pretrain.py --mode char --save char.pt --use_bf16 --no_wandb
```
**Logging with Weights & Biases (Optional):**
If you wish to log your training progress to WandB, remove the `--no_wandb` flag and specify your project/run names:
```bash
python pretrain.py --mode subword --save subword.pt --use_bf16 --project_name YourProjectName --run_name YourRunName
```

### 2. Supervised Fine-tuning
The two pre-trained backbones are integrated into a dual-branch architecture and fine-tuned for binary classification (Benign vs. DGA). Training follows a two-stage transfer learning strategy: linear probing warm-up followed by end-to-end fine-tuning.

> [!IMPORTANT]
> **Important Note on Pre-trained Weight Paths:**
> The pre-training script automatically appends the timestamp and step count to your `--save` argument. The final saved filename will look like `MMDD_HHMM_{save}_step_{steps}.pt` (e.g., `0428_1427_subword_step_2400000.pt`). **You must use this exact generated filename** when specifying the weight paths.

**Option A: Using Command-Line Arguments**
Pass the exact generated weight paths directly via the CLI:
```bash
python finetuning.py --token_weights_path 0428_1427_subword_step_2400000.pt --char_weights_path 0428_1427_char_step_2400000.pt --use_bf16 --wandb_mode disabled
```
**Option B: Configuration via `config.py`**
Alternatively, you can define the hyperparameter values and weight paths directly by modifying the `FinetuningConfig` class in `config.py`:
```python
class FinetuningConfig:
    token_weights_path: str = '0428_1427_subword_step_2400000.pt'
    char_weights_path: str = '0428_1427_char_step_2400000.pt'
    # ... other configurations (batch_size, learning_rate, etc.)
```

If you have configured the paths in `config.py`, simply run:
```bash
python finetuning.py --use_bf16 --wandb_mode disabled
```

### 3. Evaluation
Evaluation is conducted using a forward-chaining strategy, testing the model on datasets from subsequent years (e.g., 2020–2025) to strictly enforce temporal alignment and measure robustness against concept drift.

> [!IMPORTANT]
> **Important Note on Fine-tuned Model Path:**
> Similar to pre-training, the fine-tuning script automatically creates a directory and saves the model based on the execution time. The final model path will follow the `MMDD_HHMM/finetuning_MMDD_HHMM.pt` format (e.g., `0428_1430/finetuning_0428_1430.pt`). You must provide this exact path when running the evaluation.

```bash
# Evaluate on data from 2020 to 2025
python test.py --model_path 0428_1430/finetuning_0428_1430.pt --use_bf16 --no_wandb --save
```

> **Note on `--test_type`:** The default evaluation mode is `year`, which evaluates the model on the forward-chaining longitudinal datasets. Although the script also supports `--test_type family` for per-family performance analysis, the family-labeled dataset is not included in the public release. Therefore, only the default `year` mode is applicable with the provided dataset.
