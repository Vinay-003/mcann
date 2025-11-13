# MC-ANN Quick Reference Guide

## 🚀 Quick Commands

### Setup (One-Time)
```bash
# Create environment
conda create -n MCANN python=3.8.8
conda activate MCANN

# Install PyTorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Download datasets
wget https://clp.engr.scu.edu/static/datasets/MCANN-datasets.zip
unzip MCANN-datasets.zip -d data_provider/datasets/
```

### Training
```bash
# Stevens Creek (default)
python run.py --arg_file models/Stevens_Creek.txt

# Other reservoirs
python run.py --arg_file models/Coyote.txt
python run.py --arg_file models/Lexington.txt
python run.py --arg_file models/Almaden.txt
python run.py --arg_file models/Vasona.txt
```

### Inference
```bash
# Test model
python test.py --model_path "output/Stevens_Creek/train/Stevens_Creek.zip"

# Single prediction
python predict.py --model_path "output/Stevens_Creek/train/Stevens_Creek.zip" --test_time "2019-01-07 03:30:00"
```

---

## 📁 File Structure Quick Reference

```
mcann/
├── run.py                  # Main training script
├── predict.py              # Single-point prediction
├── test.py                 # Testing and visualization
├── requirements.txt        # Python dependencies
│
├── data_provider/
│   ├── DS.py              # Dataset management
│   └── datasets/          # TSV data files
│
├── models/
│   ├── GMM_Model5.py      # Encoder/Decoder architecture
│   ├── Group_GMM5.py      # Training logic (DAN class)
│   ├── Inference.py       # Standalone inference
│   └── *.txt              # Pre-configured parameters
│
├── utils/
│   ├── metric.py          # Evaluation metrics
│   └── utils2.py          # Helper functions
│
├── output/                # Generated during training
│   └── <model_name>/
│       ├── train/         # Model weights (.zip)
│       ├── val/           # Validation timestamps
│       └── test/          # Test predictions
│
└── docs/                  # Documentation (you are here!)
    ├── README.md
    ├── PROJECT_OVERVIEW.md
    ├── SETUP_AND_RUN_GUIDE.md
    ├── SYSTEM_ARCHITECTURE.md
    ├── WORKFLOW_AND_FLOWCHARTS.md
    └── PAPER_SUMMARY.md
```

---

## ⚙️ Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--reservoir_sensor` | Dataset filename | reservoir_stor_4009_sof24 | See datasets |
| `--train_volume` | Training samples | 30000 | 20K-40K |
| `--hidden_dim` | LSTM hidden size | 512 | 256-512 |
| `--atten_dim` | Attention size | 600 | 300-600 |
| `--layer` | LSTM layers | 1 | 1-4 |
| `--input_len` | Input length | 360 | 360 (15 days) |
| `--output_len` | Prediction horizon | 72 | 72 (3 days) |
| `--os_s` | Oversampling steps | 18 | 10-20 |
| `--os_v` | Oversampling freq | 4 | 4-8 |
| `--oversampling` | % extreme events | 40 | 30-50 |
| `--epochs` | Max epochs | 50 | 50-100 |
| `--learning_rate` | Initial LR | 0.001 | 0.0001-0.01 |
| `--batchsize` | Batch size | 48 | 32-64 |

---

## 🔢 Input/Output Dimensions

### Input Tensor
```
[Batch, Sequence, Features]
[48, 360, 8]

Features:
0: Normalized water level
1: Extreme score (GM3)
2-4: Dataset GMM probabilities
5-7: Sample GMM probabilities
```

### Output Tensor
```
[Batch, Horizon]
[48, 72]

72 hourly predictions (3 days ahead)
```

---

## 🏗️ Model Components

### Encoder
```python
EncoderLSTM:
  ├─ 3 Parallel LSTM Branches
  │  └─ Each: LSTM(2 → 512, layers=1-4)
  │
  ├─ Attention Module
  │  ├─ 6 MultiHeadAttention layers (4 heads each)
  │  ├─ Positional embeddings
  │  └─ BatchNorm
  │
  └─ Output: h, c, weights
```

### Decoder
```python
DecoderLSTM:
  ├─ 3 Parallel LSTM Branches
  │  └─ Each: LSTM(2 → 512), init=(h, c)
  │
  ├─ 3 Linear Layers
  │  └─ Each: Linear(512 → 1)
  │
  └─ Weighted Combination: Σ(wi × outi)
```

---

## 📊 Evaluation Metrics

```python
# RMSE (Lower is better)
RMSE = sqrt(mean((y_pred - y_true)²))

# MAPE (Lower is better)
MAPE = mean(|y_pred - y_true| / |y_true|) × 100%

# MAE (Lower is better)
MAE = mean(|y_pred - y_true|)
```

**Typical MC-ANN Performance:**
- RMSE: 40-50
- MAPE: 3-5%
- MAE: 30-40

---

## 🔄 Data Processing Pipeline

```
Raw TSV
  ↓
Parse datetime + values
  ↓
Handle NaN (tag and skip)
  ↓
Log-transform: log(x + ε)
  ↓
Standardize: (x - μ) / σ
  ↓
Compute time features (cos/sin date)
  ↓
Build GM3 (point-wise GMM)
  ↓
Build GMM0 (dataset-wise GMM)
  ↓
Sample training data (with oversampling)
  ↓
Build GMM (sample-wise GMM)
  ↓
Create DataLoader
```

---

## 🧠 Three-Level GMM

| Level | Name | Purpose | Input | Output |
|-------|------|---------|-------|--------|
| 1 | GM3 | Point-wise | All values [N×1] | Extreme scores [N×1] |
| 2 | GMM0 | Dataset-wise | 200K samples [200K×1] | Probabilities [N×3] |
| 3 | GMM | Sample-wise | Train sequences [M×72] | Probabilities [M×3] |

---

## 💾 Model Files

### Model ZIP Contents
```
model_name.zip
├── MCANN_encoder.pt    # Encoder weights
├── MCANN_decoder.pt    # Decoder weights
├── GMM.pt              # Sample-wise GMM
├── GM3.pt              # Point-wise GMM
├── GMM0.pt             # Dataset-wise GMM
├── Norm.txt            # [mean, std]
└── opt.txt             # Hyperparameters
```

### Loading Model
```python
from run import Options

opt_manager = Options()
model = opt_manager.get_model("output/Stevens_Creek/train/Stevens_Creek.zip")
```

---

## 🐛 Common Errors & Fixes

### Error: `CUDA out of memory`
```bash
# Solution 1: Reduce batch size
python run.py --batchsize 24

# Solution 2: Use CPU
python run.py --gpu_id -1
```

### Error: `FileNotFoundError: reservoir_stor_*.tsv`
```bash
# Solution: Re-download datasets
wget https://clp.engr.scu.edu/static/datasets/MCANN-datasets.zip
unzip MCANN-datasets.zip -d data_provider/datasets/
```

### Error: `ModuleNotFoundError: No module named 'sklearn'`
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

### Error: `RuntimeError: CUDA error: device-side assert triggered`
```bash
# Solution: Check data for NaN values
# Verify input tensors are in valid range
```

---

## 📈 Training Progress

### Expected Output
```
-----------Epoch: 0. train_Loss>: 145678.234. --------------------
-----------Epoch: 0. val_Loss_rmse>: 58.234. --------------------
-----------Epoch: 0. val_Loss_mape>: 0.048. --------------------
-----------Epoch: 0. running time>: 312.456. --------------------
...
-----------Epoch: 25. train_Loss>: 87654.321. --------------------
-----------Epoch: 25. val_Loss_rmse>: 47.823. --------------------
-----------Epoch: 25. val_Loss_mape>: 0.039. --------------------
...
Model saved in: ./output/Stevens_Creek/train/Stevens_Creek.zip
```

### Convergence Signs
- ✓ Training loss decreasing
- ✓ Validation RMSE stabilizing
- ✓ MAPE < 5%
- ✓ Early stopping triggered (patience=4)

---

## 🔬 Hyperparameter Tuning

### Most Important Parameters (tune first)
1. `--hidden_dim` (256, 512, 768)
2. `--atten_dim` (300, 450, 600)
3. `--layer` (1, 2, 3, 4)
4. `--oversampling` (30, 40, 50)

### Less Important (usually keep default)
- `--learning_rate` (0.001 works well)
- `--batchsize` (48 is good balance)
- `--os_s`, `--os_v` (18, 4 are reasonable)

### Grid Search Example
```bash
for hidden in 256 512 768; do
  for atten in 300 450 600; do
    python run.py --hidden_dim $hidden --atten_dim $atten --model "exp_${hidden}_${atten}"
  done
done
```

---

## 📊 Dataset Information

### Reservoirs
1. **Almaden:** reservoir_stor_4001_sof24.tsv
2. **Coyote:** reservoir_stor_4005_sof24.tsv
3. **Lexington:** reservoir_stor_4007_sof24.tsv
4. **Stevens Creek:** reservoir_stor_4009_sof24.tsv
5. **Vasona:** reservoir_stor_4011_sof24.tsv

### Time Ranges
- **Training:** 1983-07-01 to 2018-06-30 (35 years)
- **Testing:** 2018-07-01 to 2019-07-01 (1 year)
- **Resolution:** 30-minute intervals
- **Total Points:** ~630,000 per reservoir

---

## 🐍 Python API

### Training
```python
from run import Options
from data_provider.DS import DS
from models.Group_GMM5 import DAN
import pandas as pd

# Load data
trainX = pd.read_csv("data_provider/datasets/reservoir_stor_4009_sof24.tsv", sep="\t")
trainX.columns = ["datetime", "value"]

# Setup
opt = Options().parse()
ds = DS(opt, trainX)

# Train
model = DAN(opt, ds)
model.train()

# Inference
ds.refresh_dataset(trainX)
model.model_load()
predictions = model.inference()
```

### Prediction
```python
from run import Options

# Load model
opt_manager = Options()
model = opt_manager.get_model("output/Stevens_Creek/train/Stevens_Creek.zip")

# Predict
test_time = "2019-01-15 10:30:00"
predictions, ground_truth = model.test_single(test_time)

print(f"Predictions shape: {predictions.shape}")  # (72,)
print(f"First 10 predictions: {predictions[:10]}")
```

---

## 📝 Output Files

### Training Outputs
```
output/Stevens_Creek/
├── train/
│   ├── Stevens_Creek.zip     # Model package
│   ├── opt.txt                # Parameters
│   └── model.log              # Training log
│
├── val/
│   └── validation_timestamps_24avg.tsv
│
└── test/
    └── pred_lists_print.tsv  # Predictions (if --save 1)
```

### Prediction Format
```csv
timestamp,step,prediction
2019-01-07 03:30:00,0,1234.56
2019-01-07 04:30:00,1,1235.12
...
2019-01-10 02:30:00,71,1250.34
```

---

## 🎯 Best Practices

### ✓ Do
- Use GPU for training (5x faster)
- Start with pre-configured parameters
- Monitor validation metrics
- Save multiple checkpoints
- Verify data quality (no NaN in sequences)

### ✗ Don't
- Change `input_len` or `output_len` (fixed at 360/72)
- Train on data with many NaN values
- Use very small batch sizes (<16)
- Ignore early stopping signals
- Forget to activate conda environment

---

## 🔗 Useful Links

- **Paper:** https://doi.org/10.1109/TPAMI.2025.3565224
- **Dataset:** https://clp.engr.scu.edu/static/datasets/MCANN-datasets.zip
- **PyTorch Docs:** https://pytorch.org/docs/1.11.0/
- **Scikit-learn GMM:** https://scikit-learn.org/stable/modules/mixture.html

---

## 📞 Quick Help

| Issue | Solution |
|-------|----------|
| Slow training | Use GPU (`--gpu_id 0`) |
| Out of memory | Reduce batch size (`--batchsize 24`) |
| High validation RMSE | Increase `--train_volume` or `--epochs` |
| Poor extreme prediction | Increase `--oversampling` |
| NaN in predictions | Check input data quality |

---

## 📚 Related Documentation

- **Full Setup:** [SETUP_AND_RUN_GUIDE.md](./SETUP_AND_RUN_GUIDE.md)
- **Architecture:** [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)
- **Workflows:** [WORKFLOW_AND_FLOWCHARTS.md](./WORKFLOW_AND_FLOWCHARTS.md)
- **Paper Analysis:** [PAPER_SUMMARY.md](./PAPER_SUMMARY.md)

---

**Last Updated:** November 14, 2025  
**Version:** 1.0
