# CDA-CycleGAN: Cross-Domain Attention for 3T-to-7T MRI Synthesis

This folder contains the core Python scripts to:

- Train the model (`Train.py`)
- Run reconstruction/inference (`reconstruct.py`)
- Compute image quality metrics between generated and real images (`extract_matrices.py`)

## Prerequisites

- Python 3.10+ recommended
- (Optional) NVIDIA GPU with CUDA for faster training/inference

## 1) Create and activate a virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

If you want GPU-enabled PyTorch, install PyTorch according to the official instructions for your CUDA version, then install the remaining dependencies:

- PyTorch install selector: `https://pytorch.org/get-started/locally/`

## 3) Prepare your datasets (folder structure)

The code expects the following local folders (you can rename them, but then update the variables inside the scripts accordingly):

### Training dataset

Create:

```
training_dataset/
  3T/
    <images...>
  7T/
    <images...>
```

`Train.py` uses:

- `TRAIN_DIR = "training_dataset"`
- `root_a = TRAIN_DIR + "/3T"`
- `root_b = TRAIN_DIR + "/7T"`

### Testing datasets

For reconstruction:

```
3T_testing_dataset/
  <images...>
```

For metric extraction (real 7T test set):

```
7T_testing_dataset/
  <images...>
```

## 4) Train

From this folder:

```bash
python Train.py
```

Outputs:

- A `Results/` directory is created (if missing)
- Checkpoints are written periodically inside `Results/`
- Generated sample images are saved under:
  - `Results/Generated from 7T/`
  - `Results/Generated from 3T/`

Important variables you may want to adjust (near the bottom of `Train.py`):

- `NUM_EPOCHS`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `IMAGE_HEIGHT`, `IMAGE_WIDTH`

The script automatically selects device as:

- CUDA if available, else CPU

## 5) Reconstruct / inference

Edit these variables at the top of `reconstruct.py` if needed:

- `input_dir = "3T_testing_dataset"`
- `output_dir = "fake_dataset_7T_780"`
- `checkpoint = "Results/gen_B_epoch_780.pth"`

Then run:

```bash
python reconstruct.py
```

Output:

- Generated images saved under `output_dir`

## 6) Compute metrics (PSNR / MSE / RMSE / SSIM)

Edit at the bottom of `extract_matrices.py` if needed:

- `fake_folder = "fake_dataset_7T_780"`
- `real_folder = "7T_testing_dataset"`
- `output_csv = "image_comparison_results_780.csv"`

Run:

```bash
python extract_matrices.py
```

Output:

- A CSV file (`output_csv`) with per-image metrics.

## Notes

- All scripts are written to run on either CUDA or CPU.
- Paths are intentionally kept as short local placeholders so this folder can be uploaded to GitHub and run on another machine.

