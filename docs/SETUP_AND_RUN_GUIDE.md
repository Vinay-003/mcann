# MC-ANN Setup and Execution Guide

## Complete Setup Instructions

Follow these steps to set up and run MC-ANN on your system.

---

## Step 1: Environment Setup

### 1.1 Create Conda Environment

```bash
# Create a new conda environment with Python 3.8.8
conda create -n MCANN python=3.8.8

# Activate the environment
conda activate MCANN
```

### 1.2 Install PyTorch

```bash
# Install PyTorch 1.11.0 (CUDA version - if you have a GPU)
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch

# OR for CPU-only installation
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
```

### 1.3 Install Dependencies

```bash
# Navigate to the project directory
cd /home/mylappy/Desktop/mcann

# Install required Python packages
python -m pip install -r requirements.txt
```

**Dependencies installed:**
- matplotlib==3.7.4
- numpy==1.24.4
- pandas==2.0.3
- scikit-learn==1.1.2
- scipy==1.10.1
- unicodecsv==0.14.1
- threadpoolctl==3.1.0
- jupyter

---

## Step 2: Download Datasets

### 2.1 Download Data Files

```bash
# Download the dataset zip file
wget https://clp.engr.scu.edu/static/datasets/MCANN-datasets.zip

# Or use curl if wget is not available
curl -O https://clp.engr.scu.edu/static/datasets/MCANN-datasets.zip
```

### 2.2 Extract Datasets

```bash
# Unzip the files into the data_provider/datasets directory
unzip MCANN-datasets.zip -d data_provider/datasets/

# Verify the files are extracted
ls data_provider/datasets/
```

**Expected files:**
- `reservoir_stor_4001_sof24.tsv` (Almaden)
- `reservoir_stor_4005_sof24.tsv` (Coyote)
- `reservoir_stor_4007_sof24.tsv` (Lexington)
- `reservoir_stor_4009_sof24.tsv` (Stevens Creek)
- `reservoir_stor_4011_sof24.tsv` (Vasona)
- `test_timestamps_24avg.tsv` (Test timestamps)

---

## Step 3: Training the Model

### 3.1 Quick Start: Train Stevens Creek Model

```bash
# Activate the conda environment (if not already activated)
conda activate MCANN

# Train the model with default parameters for Stevens Creek
python run.py \
  --train_volume 30000 \
  --hidden_dim 512 \
  --atten_dim 600 \
  --layer 1 \
  --reservoir_sensor reservoir_stor_4009_sof24 \
  --os_s 18 \
  --os_v 4 \
  --seq_weight 0.4 \
  --oversampling 40 \
  --input_len 360 \
  --output_len 72 \
  --epochs 50 \
  --model Stevens_Creek
```

**Training Progress:**
- Model creates `./output/Stevens_Creek/train/` directory
- Saves validation points to `./output/Stevens_Creek/val/`
- Checkpoints saved as `.zip` files containing model weights

**Expected Output:**
```
-----------Epoch: 0. train_Loss>: 123456.789. --------------------
-----------Epoch: 0. val_Loss_rmse>: 45.678. --------------------
-----------Epoch: 0. val_Loss_mape>: 0.034. --------------------
...
Model saved in: ./output/Stevens_Creek/train/Stevens_Creek.zip
```

### 3.2 Train Other Reservoirs

**Coyote Reservoir:**
```bash
python run.py \
  --arg_file models/Coyote.txt \
  --model Coyote
```

**Lexington Reservoir:**
```bash
python run.py \
  --arg_file models/Lexington.txt \
  --model Lexington
```

**Almaden Reservoir:**
```bash
python run.py \
  --arg_file models/Almaden.txt \
  --model Almaden
```

**Vasona Reservoir:**
```bash
python run.py \
  --arg_file models/Vasona.txt \
  --model Vasona
```

### 3.3 Training Parameters Explained

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `--reservoir_sensor` | Dataset filename (without .tsv) | See datasets |
| `--train_volume` | Number of training samples | 20,000-40,000 |
| `--hidden_dim` | LSTM hidden dimension | 256-512 |
| `--atten_dim` | Attention dimension | 300-600 |
| `--layer` | Number of LSTM layers | 1-4 |
| `--os_s` | Oversampling steps | 10-20 |
| `--os_v` | Oversampling frequency | 4-8 |
| `--seq_weight` | Sequence cluster weight | 0.3-0.5 |
| `--oversampling` | % of extreme events in training | 30-50 |
| `--input_len` | Input sequence length | 360 (15 days) |
| `--output_len` | Prediction horizon | 72 (3 days) |
| `--epochs` | Maximum training epochs | 50-100 |
| `--learning_rate` | Initial learning rate | 0.001 |
| `--batchsize` | Training batch size | 32-64 |

---

## Step 4: Model Inference

### 4.1 Test on Test Set

After training, run inference on the entire test set:

```bash
python test.py \
  --model_path "output/Stevens_Creek/train/Stevens_Creek.zip" \
  --test_time "2018-07-04 08:30:00"
```

**Output:**
- Generates predictions vs. ground truth plot
- Saves figure as `output.png`
- Displays RMSE and MAPE metrics

### 4.2 Single Point Prediction

Predict at a specific timestamp:

```bash
python predict.py \
  --model_path "output/Stevens_Creek/train/Stevens_Creek.zip" \
  --test_time "2019-01-07 03:30:00"
```

**Output:**
- Saves predictions to `predict.txt`
- Contains 72 hourly predictions (3 days ahead)

### 4.3 Programmatic Inference (Python)

```python
from run import Options

# Load trained model
opt_manager = Options()
model = opt_manager.get_model("output/Stevens_Creek/train/Stevens_Creek.zip")

# Predict at a specific time point
test_time = "2019-01-15 10:30:00"
predictions, ground_truth = model.test_single(test_time)

print(f"Predictions: {predictions}")
print(f"Ground Truth: {ground_truth}")
```

---

## Step 5: Jupyter Notebook Tutorial

### 5.1 Launch Jupyter

```bash
# Activate the environment
conda activate MCANN

# Start Jupyter Notebook
jupyter notebook
```

### 5.2 Open Tutorial

Open `example.ipynb` in your browser:
- Cell 1: Parameter explanation
- Cell 2: Train Stevens Creek model
- Cell 3: Test the model
- Cell 4: Single-point prediction

### 5.3 Run Experiments

Use `experiments.ipynb` for:
- Custom experiments
- Hyperparameter tuning
- Result visualization
- Comparative analysis

---

## Step 6: Validate Paper Results

### 6.1 Reproduce Paper Results

To validate the results from the TPAMI 2025 paper:

```bash
# Train all 5 reservoir models
for reservoir in Almaden Coyote Lexington Stevens_Creek Vasona; do
  python run.py --arg_file models/${reservoir}.txt --model ${reservoir}
done

# Generate test set predictions
for reservoir in Almaden Coyote Lexington Stevens_Creek Vasona; do
  python test.py --model_path "output/${reservoir}/train/${reservoir}.zip"
done
```

### 6.2 Evaluation Metrics

The model outputs three key metrics:

1. **RMSE (Root Mean Square Error)**: Average prediction error
2. **MAE (Mean Absolute Error)**: Absolute prediction error
3. **MAPE (Mean Absolute Percentage Error)**: Relative error percentage

**Expected Performance (Paper Results):**
- RMSE: 30-60 units (varies by reservoir)
- MAPE: 2-5% (3-day ahead forecasts)

### 6.3 Compare Results

Check your results against the paper:
- Results are saved in `./output/<model_name>/test/`
- Predictions saved as `pred_lists_print.tsv`
- Compare RMSE/MAPE with Table 2-4 in the paper

---

## Step 7: Common Issues and Solutions

### 7.1 CUDA Out of Memory

**Problem**: GPU memory exhausted during training

**Solution**:
```bash
# Reduce batch size
python run.py --batchsize 24 ...

# Or use CPU
python run.py --gpu_id -1 ...
```

### 7.2 Missing Dataset Files

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**:
```bash
# Verify dataset location
ls data_provider/datasets/

# Re-download if missing
wget https://clp.engr.scu.edu/static/datasets/MCANN-datasets.zip
unzip MCANN-datasets.zip -d data_provider/datasets/
```

### 7.3 Import Errors

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check environment
conda list
```

### 7.4 PyTorch Version Mismatch

**Problem**: Model fails to load due to PyTorch version

**Solution**:
```bash
# Reinstall exact PyTorch version
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
```

---

## Step 8: Output File Structure

After training and inference, the output directory structure:

```
output/
└── Stevens_Creek/
    ├── train/
    │   ├── Stevens_Creek.zip     # Trained model (encoder + decoder + GMMs)
    │   ├── opt.txt                # Training parameters
    │   └── model.log              # Training logs
    ├── val/
    │   └── validation_timestamps_24avg.tsv  # Validation time points
    └── test/
        └── pred_lists_print.tsv   # Test set predictions (if --save 1)
```

**Model ZIP Contents:**
- `MCANN_encoder.pt`: Encoder weights
- `MCANN_decoder.pt`: Decoder weights
- `GMM.pt`: Sample-wise GMM
- `GM3.pt`: Point-wise GMM
- `GMM0.pt`: Dataset-wise GMM
- `Norm.txt`: Normalization parameters
- `opt.txt`: Model hyperparameters

---

## Step 9: Performance Optimization

### 9.1 GPU Acceleration

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Use specific GPU
python run.py --gpu_id 0 ...  # Use GPU 0

# Use multiple GPUs
python run.py --ngpu 2 ...    # Use 2 GPUs
```

### 9.2 Training Speed

- **Batch size**: Increase to 64+ if GPU memory allows
- **Workers**: Increase num_workers in DataLoader (currently 2)
- **Mixed precision**: Not implemented, but can be added for 2x speedup

### 9.3 Inference Speed

- **Batch inference**: Process multiple time points together
- **Model quantization**: Use PyTorch quantization for faster inference
- **ONNX export**: Export to ONNX for deployment

---

## Step 10: Next Steps

### After successful setup:

1. **Explore the Data**: Visualize time series patterns in `experiments.ipynb`
2. **Hyperparameter Tuning**: Adjust parameters for your specific use case
3. **Custom Datasets**: Adapt the code for your own time series data
4. **Model Comparison**: Compare with baseline methods (ARIMA, LSTM, Transformer)
5. **Deployment**: Package the model for production use

### Resources:

- **Paper**: Read the full TPAMI 2025 paper for theoretical details
- **Code Documentation**: Check inline comments in source files
- **Examples**: Study `example.ipynb` for usage patterns
- **Issues**: Report bugs or ask questions on the repository

---

## Summary Checklist

- [ ] Conda environment created and activated
- [ ] PyTorch and dependencies installed
- [ ] Datasets downloaded and extracted
- [ ] Training completed successfully
- [ ] Model saved to output directory
- [ ] Inference tested on test set
- [ ] Results validated against paper
- [ ] Documentation reviewed

**Congratulations! You're now ready to use MC-ANN for time series forecasting!**
