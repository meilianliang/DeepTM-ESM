# High-Throughput Prediction of Protein Thermostability Using ESM-2, Hand-crafted Features, and Self-Attentive GCN
## Abstract:

Predicting protein thermal stability is critical for enzyme engineering, yet existing models often involve complex, multi-step pipelines that depend on costly alignments or predicted structures, severely limiting their applicability for high-throughput screening. We present DeepTM-ESM, a lightweight framework combining ESM-2 protein language model embeddings with hand-crafted biophysical features. These features are processed by a graph convolutional network augmented with a self-attention mechanism to capture both local residue interactions and global sequence context. Two variants (with/without optimal growth temperature, OGT) are introduced.



## Installation

Ensure Python 3.8+ is installed, then run:

```sh
chmod +x *.sh
./install.sh
```



## Usage

### 1. Training

We provide two scripts for models **with OGT** (`train.sh`) and **without OGT** (`train_noogt.sh`). Both accept the same arguments:

```bash
./train.sh <train_data.csv> [mode]       # for models with OGT
./train_noogt.sh <train_data.csv> [mode] # for models without OGT
```

**Arguments:**
- `<train_data.csv>`: Path to training CSV file (must be inside project root).
- `[mode]` (optional): Select steps to run. Default = `3`.
  - `1`: Run steps 2–4 (skip ESM embedding extraction)
  - `2`: Run step 4 only (training)
  - `3`: Run all steps (1–4)

**Step details:**
- Step 1: ESM-2 embedding extraction (`gen_pt.py`)
- Step 2: Node feature creation (`get_features.py`)
- Step 3: Edge feature creation (`gcm.py`)
- Step 4: Training (`main_train_with_valid.py`)

**Special case for DeepSTABp dataset:**
If you are training on the DeepSTABp dataset (contains sequences with special characters and maximum length 1750), set the environment variable `DATASET=DeepSTABp` to enable specific preprocessing and training configurations:

```bash
DATASET=DeepSTABp ./train.sh <train_data.csv> [mode]
DATASET=DeepSTABp ./train_noogt.sh <train_data.csv> [mode]
```

**Examples:**
```bash
./train.sh Tm50Train.csv           # run all steps (OGT model)
./train_noogt.sh Tm50Train.csv 1   # run steps 2–4 (non-OGT model)
DATASET=DeepSTABp ./train.sh deepstabp_train.csv  # DeepSTABp dataset with OGT
```

**Note:** Trained model weights will be saved in `DeepTM-ESM/Model/` by default. The training script automatically saves the best model during training. 



### 2. Testing

We provide two scripts for models **with OGT** (`test.sh`) and **without OGT** (`test_noogt.sh`). Both accept the same arguments:

```bash
./test.sh <test_data.csv> [mode]       # for models with OGT
./test_noogt.sh <test_data.csv> [mode] # for models without OGT
```

**Arguments:**
- `<test_data.csv>`: Path to test CSV file (must be inside project root).
- `[mode]` (optional): Select steps to run. Default = `3`.
  - `1`: Run steps 2–4 (skip ESM embedding extraction)
  - `2`: Run step 4 only (testing)
  - `3`: Run all steps (1–4)

**Special case for DeepSTABp dataset:**
If you are testing on the DeepSTABp dataset (contains sequences with special characters and maximum length 1750), set the environment variable DATASET=DeepSTABp to enable specific preprocessing and testing configurations:

```bash
DATASET=DeepSTABp ./test.sh <test_data.csv> [mode]
DATASET=DeepSTABp ./test_noogt.sh <test_data.csv> [mode]
```

**Examples:**

```bash
./test.sh Tm50Test.csv           # run all steps (OGT model)
./test_noogt.sh Tm50Test.csv 1   # run steps 2–4 (non-OGT model)
DATASET=DeepSTABp ./test.sh deepstabp_test.csv  # DeepSTABp dataset with OGT
```

**Note:** Place your trained model files in `DeepTM-ESM/Model/`. The test script will automatically use all models in this directory for evaluation.



1. ### 3. Prediction

   Use the `predict.sh` script to apply a trained model to new sequences and generate predictions.

   ```bash
   ./predict.sh --input <test.csv> --model <model.pkl> --output <results.csv> (--ogt|--no-ogt) [--mode <1|2|3>]
   ```

   **Required arguments:**
   - `--input PATH`      : Path to input CSV file.
   - `--model PATH`      : Path to trained model file (`.pkl` format).
   - `--output PATH`     : Path to output CSV file where predictions will be saved.
   - `--ogt` / `--no-ogt`: Specify whether the model was trained **with OGT** (`--ogt`) or **without OGT** (`--no-ogt`). This determines which prediction backend is used.

   **Input file format:**
   The input CSV must contain the following columns:
   - `uniprot_id` : Unique identifier for each sequence.
   - `sequence`   : The amino acid sequence.

   If you use the `--ogt` option (model trained with OGT), the CSV must **also** contain a column named `ogt` with the optimal growth temperature values for each sequence.

   For reference, see the example file `samples.csv` in the project root.

   **Optional arguments:**
   - `--mode NUM`        : Execution mode (default = `3`).
     - `1` : Run steps 2–4 (skip ESM‑2 embedding extraction).
     - `2` : Run step 4 only (prediction).
     - `3` : Run all steps (1–4).
   - `-h, --help`        : Show help message.

   **Step details:**
   - Step 1: ESM‑2 embedding extraction (`gen_pt.py`)
   - Step 2: Node feature creation (`get_features_test.py`)
   - Step 3: Edge feature creation (`gcm.py`)
   - Step 4: Prediction (`predict.py`)

   **Dataset auto‑detection:**
   If the model filename contains `DeepSTABp` (case‑insensitive), the script automatically applies special preprocessing required for the DeepSTABp dataset (sequences with special characters and max length 1750).

   **Required parameter files:**
   The following `.npy` files must exist **in the same directory as the model file**. They will be copied to `DeepTM-ESM/Data/` automatically.
   - `mean_noblhhm.npy`
   - `mean_ogt.npy`
   - `std_noblhhm.npy`
   - `std_ogt.npy`

   **Examples:**
   ```bash
   # Predict using an OGT model, all steps (mode 3)
   ./predict.sh --input new_seqs.csv --model Pre-trained/TmPred/TmPred_ogt.pkl --output predictions.csv --ogt
   
   # Predict using a non‑OGT model, skip ESM‑2 embedding (mode 1)
   ./predict.sh --input new_seqs.csv --model Pre-trained/TmPred/TmPred_noogt.pkl --output preds.csv --no-ogt --mode 1
   
   # DeepSTABp model (filename contains "DeepSTABp") triggers special preprocessing automatically
   ./predict.sh --input deep_data.csv --model Pre-trained/DeepSTABp/DeepSTABp_ogt.pkl --output deep_preds.csv --ogt
   ```



### 4. Datasets and Pre-trained Models

#### Datasets

The project includes four benchmark datasets for training and evaluation. All dataset files are located in the `DeepTM-ESM/` root directory.

| Dataset       | Description                                                  | Source                                       |
| ------------- | ------------------------------------------------------------ | -------------------------------------------- |
| **Tm50**      | 7,790 sequences (50–100°C) from the Meltome Atlas            | https://github.com/liimy1/DeepTM             |
| **DeepSTABp** | 35,112 sequences (30–90°C) from Meltome Atlas and TPP data, deduplicated to 28,015 unique sequences | https://git.nfdi4plants.org/f_jung/deepstabp |
| **TmPred**    | 3,542 sequences (>60°C) from the Meltome Atlas               | https://github.com/alwaysniu/TmPred          |
| **Blind**     | Independent blind test set for final evaluation              |                                              |

**Dataset files:**
```
DeepTM-ESM/
├── Tm50Train.csv           # Tm50 training set
├── Tm50Test.csv            # Tm50 test set
├── deepstabp_train.csv     # DeepSTABp training set
├── deepstabp_test.csv      # DeepSTABp test set
├── TmpredTrain.csv         # TmPred training set
├── TmpredTest.csv          # TmPred test set
└── Blind.csv               # Blind test set
```

#### Pre-trained Models

Pre-trained models and their associated parameter files are organized in the `Pre-trained/` directory. Each dataset has its own subdirectory containing two model variants (with and without OGT).

**Directory structure:**
```
Pre-trained/
├── Tm50/
│   ├── Tm50_ogt.pkl           # OGT model trained on Tm50
│   ├── Tm50_noogt.pkl         # Non-OGT model trained on Tm50
│   ├── mean_noblhhm.npy
│   ├── mean_ogt.npy
│   ├── std_noblhhm.npy
│   └── std_ogt.npy
├── DeepSTABp/
│   ├── DeepSTABp_ogt.pkl
│   ├── DeepSTABp_noogt.pkl
│   └── [parameter files...]
└── TmPred/
    ├── TmPred_ogt.pkl
    ├── TmPred_noogt.pkl
    └── [parameter files...]

```

**Important notes:**
- **Directory naming**: The subdirectory name under `Pre-trained/`  exactly match the dataset name (case-sensitive) for automatic dataset detection.
- **Model files**: Each model file (`.pkl`) is accompanied by the four parameter files (`mean_noblhhm.npy`, `mean_ogt.npy`, `std_noblhhm.npy`, `std_ogt.npy`) in the same directory.
- **Dataset detection**: When using a model from a dataset-specific directory (e.g., `Pre-trained/DeepSTABp/`), the corresponding preprocessing will be automatically applied (e.g., DeepSTABp-specific handling for special characters and max length 1750).

**Usage example:**

```bash
# Use DeepSTABp model with OGT
./predict.sh --input deepstabp_test.csv \
             --model Pre-trained/DeepSTABp/DeepSTABp_model.pkl \
             --output predictions.csv \
             --ogt
```
