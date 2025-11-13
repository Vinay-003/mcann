# MC-ANN System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MC-ANN SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  Data Layer   │→→│ Model Layer  │→→│  Inference Layer │    │
│  └───────────────┘  └──────────────┘  └──────────────────┘    │
│         ↓                  ↓                    ↓               │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ Preprocessing │  │   Training   │  │    Prediction    │    │
│  └───────────────┘  └──────────────┘  └──────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Layer Architecture

### 1.1 Data Pipeline

```
Raw TSV Files → Data Loading → Preprocessing → Feature Engineering → Training Batches
     ↓              ↓               ↓                  ↓                  ↓
[Datasets]     [DS.py]      [Normalization]    [GMM Features]      [DataLoader]
```

### 1.2 Data Provider (DS.py)

**Components:**
- **Data Reader**: Loads TSV files with datetime and water level values
- **Normalizer**: Log-standard normalization with differencing
- **Feature Generator**: Creates temporal and clustering features
- **Sampler**: Random sampling with oversampling for extreme events
- **DataLoader**: PyTorch DataLoader for batch training

**Key Methods:**
```python
read_dataset()          # Load and preprocess raw data
train_dataloader()      # Generate training batches
val_dataloader()        # Generate validation samples
refresh_dataset()       # Update for new data (inference)
gen_test_data()         # Create test set
```

### 1.3 Input Features (8-dimensional)

```
┌──────────────────────────────────────────────────────────────┐
│ Input Tensor: [Batch, Sequence_Length=360, Features=8]       │
├──────────────────────────────────────────────────────────────┤
│ Dim 0: Normalized water level (log-std normalized)           │
│ Dim 1: Extreme score (from GM3 - point-wise GMM)            │
│ Dim 2: Dataset GMM component 1 probability (from GMM0)       │
│ Dim 3: Dataset GMM component 2 probability (from GMM0)       │
│ Dim 4: Dataset GMM component 3 probability (from GMM0)       │
│ Dim 5: Sample GMM component 1 probability (from GMM)         │
│ Dim 6: Sample GMM component 2 probability (from GMM)         │
│ Dim 7: Sample GMM component 3 probability (from GMM)         │
└──────────────────────────────────────────────────────────────┘
```

### 1.4 Data Augmentation Strategy

**Oversampling Extreme Events:**
```python
# Algorithm:
1. Identify extreme sequences (max > threshold OR min < threshold)
2. Sample around the extreme point (±18 steps × 4 frequency)
3. Maintain 30-40% extreme events in training set
4. Ensures model learns rare but important patterns
```

---

## 2. Model Layer Architecture

### 2.1 Overall Model Structure

```
                    ┌──────────────────────────────────┐
                    │       Input Sequence             │
                    │   [Batch, 360, 8 features]      │
                    └──────────┬───────────────────────┘
                               │
                               ↓
                    ┌──────────────────────────────────┐
                    │      ENCODER MODULE              │
                    ├──────────────────────────────────┤
                    │  ┌────────────────────────────┐  │
                    │  │   3 Parallel LSTM Branches│  │
                    │  │   (for 3 GMM components)  │  │
                    │  └────────────────────────────┘  │
                    │              ↓                    │
                    │  ┌────────────────────────────┐  │
                    │  │   Multi-Head Attention    │  │
                    │  │   (6 attention modules)   │  │
                    │  └────────────────────────────┘  │
                    │              ↓                    │
                    │  [Hidden States h0, h1, h2]     │
                    │  [Cell States c0, c1, c2]       │
                    │  [Attention Weights w0, w1, w2] │
                    └──────────┬───────────────────────┘
                               │
                               ↓
                    ┌──────────────────────────────────┐
                    │      DECODER MODULE              │
                    ├──────────────────────────────────┤
                    │  ┌────────────────────────────┐  │
                    │  │   3 Parallel LSTM Branches│  │
                    │  │   (initialized with h, c) │  │
                    │  └────────────────────────────┘  │
                    │              ↓                    │
                    │  ┌────────────────────────────┐  │
                    │  │   Weighted Combination    │  │
                    │  │   out = Σ(wi × outi)      │  │
                    │  └────────────────────────────┘  │
                    │              ↓                    │
                    │    [Predictions: Batch, 72]     │
                    └──────────────────────────────────┘
```

### 2.2 Encoder Architecture (EncoderLSTM)

**Layer Details:**

```python
EncoderLSTM:
  ├─ Input: [Batch, 360, 2]  # Only dim 0 and 1 used directly
  │
  ├─ LSTM Branch 0: [Batch, 360, 2] → [Batch, 360, 512]
  │  └─ LSTM(input=2, hidden=512, layers=1-4, dropout=0.1)
  │
  ├─ LSTM Branch 1: [Batch, 360, 2] → [Batch, 360, 512]
  │  └─ LSTM(input=2, hidden=512, layers=1-4, dropout=0.1)
  │
  ├─ LSTM Branch 2: [Batch, 360, 2] → [Batch, 360, 512]
  │  └─ LSTM(input=2, hidden=512, layers=1-4, dropout=0.1)
  │
  ├─ Attention Module (for clustering weights):
  │  ├─ Input: GMM probabilities [Batch, 72, 3]
  │  │
  │  ├─ Component 0 Processing:
  │  │  ├─ Linear: [B, 72, 1] → [B, 72, 150]
  │  │  ├─ Add Positional Embedding: [B, 72, 150]
  │  │  ├─ Self-Attention 1: MultiHeadAttention(150, 4 heads)
  │  │  ├─ Add & Norm: BatchNorm1d
  │  │  ├─ Self-Attention 2: MultiHeadAttention(150, 4 heads)
  │  │  ├─ Add & Norm: BatchNorm1d
  │  │  └─ Linear: [B, 72, 150] → [B, 72, 1]
  │  │
  │  ├─ Component 1 Processing: (same as Component 0)
  │  ├─ Component 2 Processing: (same as Component 0)
  │  │
  │  ├─ Concatenate: [B, 72, 3]
  │  └─ Softmax: Σ wi = 1 for each timestep
  │
  └─ Output:
     ├─ h: List of 3 hidden states [3 × [layers, Batch, 512]]
     ├─ c: List of 3 cell states [3 × [layers, Batch, 512]]
     └─ ww: Attention weights [Batch, 72, 3]
```

### 2.3 Decoder Architecture (DecoderLSTM)

```python
DecoderLSTM:
  ├─ Input:
  │  ├─ Temporal features: [Batch, 72, 2]  # cos/sin date encoding
  │  ├─ Hidden states: [3 × [layers, Batch, 512]]
  │  ├─ Cell states: [3 × [layers, Batch, 512]]
  │  └─ Attention weights: [Batch, 72, 3]
  │
  ├─ LSTM Branch 0:
  │  ├─ LSTM(input=2, hidden=512, init=(h0, c0))
  │  ├─ Output: [Batch, 72, 512]
  │  └─ Linear: [Batch, 72, 512] → [Batch, 72]
  │
  ├─ LSTM Branch 1:
  │  ├─ LSTM(input=2, hidden=512, init=(h1, c1))
  │  ├─ Output: [Batch, 72, 512]
  │  └─ Linear: [Batch, 72, 512] → [Batch, 72]
  │
  ├─ LSTM Branch 2:
  │  ├─ LSTM(input=2, hidden=512, init=(h2, c2))
  │  ├─ Output: [Batch, 72, 512]
  │  └─ Linear: [Batch, 72, 512] → [Batch, 72]
  │
  └─ Weighted Combination:
     ├─ out = w0 ⊙ out0 + w1 ⊙ out1 + w2 ⊙ out2
     └─ Output: [Batch, 72]  # Final predictions
```

### 2.4 Gaussian Mixture Models (GMM)

**Three-Level GMM Hierarchy:**

```
┌─────────────────────────────────────────────────────────────┐
│ Level 1: GM3 - Point-wise Clustering                        │
├─────────────────────────────────────────────────────────────┤
│ Input: All non-NaN normalized values [N, 1]                 │
│ Output: Extreme score (1 - P(normal))                       │
│ Purpose: Identify individual extreme values                 │
│ Components: 3 (low, medium, high)                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Level 2: GMM0 - Dataset-wise Clustering                     │
├─────────────────────────────────────────────────────────────┤
│ Input: Random samples of last 72 values [200K, 1]          │
│ Output: Posterior probabilities P(component | data)         │
│ Purpose: Capture dataset-level patterns                     │
│ Components: 3 (ordered by weight: high, low, medium)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Level 3: GMM - Sample-wise Clustering                       │
├─────────────────────────────────────────────────────────────┤
│ Input: Last 72 extreme scores of training samples [M, 72]  │
│ Output: Sequence-level probabilities                        │
│ Purpose: Classify entire sequences by extreme patterns     │
│ Components: 3 (ordered by weight: max, min, medium)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Training Layer Architecture

### 3.1 Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  For each epoch:                                             │
│    ┌──────────────────────────────────────────┐            │
│    │ 1. Forward Pass                          │            │
│    │    • Encoder: x → (h, c, w)             │            │
│    │    • Decoder: (y_date, h, c, w) → pred │            │
│    └──────────────────────────────────────────┘            │
│                    ↓                                         │
│    ┌──────────────────────────────────────────┐            │
│    │ 2. Denormalization                       │            │
│    │    • pred_denorm = pred * std + mean    │            │
│    │    • pred_final = pred_denorm + y_prev  │            │
│    └──────────────────────────────────────────┘            │
│                    ↓                                         │
│    ┌──────────────────────────────────────────┐            │
│    │ 3. Loss Calculation                      │            │
│    │    • loss = MSE(pred_final, y_true)     │            │
│    └──────────────────────────────────────────┘            │
│                    ↓                                         │
│    ┌──────────────────────────────────────────┐            │
│    │ 4. Backpropagation                       │            │
│    │    • loss.backward()                     │            │
│    │    • optimizer.step()                    │            │
│    └──────────────────────────────────────────┘            │
│                    ↓                                         │
│    ┌──────────────────────────────────────────┐            │
│    │ 5. Validation                            │            │
│    │    • Test on 60 validation points        │            │
│    │    • Compute RMSE & MAPE                │            │
│    │    • Save best model                     │            │
│    └──────────────────────────────────────────┘            │
│                    ↓                                         │
│    ┌──────────────────────────────────────────┐            │
│    │ 6. Early Stopping Check                  │            │
│    │    • If val_loss increases for 4 epochs │            │
│    │    • Stop training                       │            │
│    └──────────────────────────────────────────┘            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Training Controller (Group_GMM5.py)

**DAN Class (Deep Attention Network):**

```python
DAN:
  ├─ __init__(opt, dataset)
  │  ├─ Initialize encoder & decoder
  │  ├─ Setup optimizers (Adam)
  │  ├─ Define loss functions (MSE, Huber, KL-Divergence)
  │  └─ Load training/validation data
  │
  ├─ train()
  │  ├─ For each epoch:
  │  │  ├─ Training loop:
  │  │  │  ├─ Forward pass
  │  │  │  ├─ Compute loss
  │  │  │  ├─ Backward pass
  │  │  │  └─ Update weights
  │  │  ├─ Validation loop:
  │  │  │  ├─ Test on validation set
  │  │  │  ├─ Compute RMSE/MAPE
  │  │  │  └─ Save if best model
  │  │  └─ Adjust learning rate
  │  └─ Return trained model
  │
  ├─ inference()
  │  ├─ Load trained model
  │  ├─ Generate test predictions
  │  └─ Save results
  │
  ├─ test_single(test_point)
  │  ├─ Extract input sequence
  │  ├─ Forward pass
  │  └─ Return predictions
  │
  └─ model_load()
     └─ Load encoder/decoder from ZIP file
```

---

## 4. Inference Layer Architecture

### 4.1 Inference Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                   Inference Flow                              │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────┐                                         │
│  │ Load Model ZIP  │                                         │
│  │ • encoder.pt    │                                         │
│  │ • decoder.pt    │                                         │
│  │ • GMM models    │                                         │
│  │ • opt.txt       │                                         │
│  └────────┬────────┘                                         │
│           ↓                                                   │
│  ┌─────────────────┐                                         │
│  │ Load Data       │                                         │
│  │ • TSV file      │                                         │
│  │ • Normalize     │                                         │
│  └────────┬────────┘                                         │
│           ↓                                                   │
│  ┌─────────────────┐                                         │
│  │ Extract Inputs  │                                         │
│  │ • Last 360 pts  │                                         │
│  │ • Compute GMM   │                                         │
│  │ • Time features │                                         │
│  └────────┬────────┘                                         │
│           ↓                                                   │
│  ┌─────────────────┐                                         │
│  │ Forward Pass    │                                         │
│  │ • Encoder       │                                         │
│  │ • Decoder       │                                         │
│  └────────┬────────┘                                         │
│           ↓                                                   │
│  ┌─────────────────┐                                         │
│  │ Denormalize     │                                         │
│  │ • Inverse norm  │                                         │
│  │ • Add previous  │                                         │
│  └────────┬────────┘                                         │
│           ↓                                                   │
│  ┌─────────────────┐                                         │
│  │ Output          │                                         │
│  │ • 72 values     │                                         │
│  │ • 3 days ahead  │                                         │
│  └─────────────────┘                                         │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Inference Module (Inference.py)

```python
MCANN_I:
  ├─ __init__(opt)
  │  ├─ Load hyperparameters
  │  └─ Initialize encoder/decoder
  │
  ├─ model_load(pt_file)
  │  ├─ Extract from ZIP
  │  ├─ Load encoder weights
  │  ├─ Load decoder weights
  │  └─ Load GMM models
  │
  ├─ predict(test_time)
  │  ├─ Prepare input sequence
  │  ├─ Run inference
  │  └─ Return predictions
  │
  └─ batch_predict(test_times)
     └─ Process multiple timestamps
```

---

## 5. Utility Layer Architecture

### 5.1 Preprocessing Utilities (utils2.py)

```python
Utility Functions:
├─ log_std_normalization(data)
│  └─ (log(x+ε) - mean) / std
│
├─ r_log_std_normalization(data)
│  └─ Reverse: exp(x * std + mean) - ε
│
├─ diff_order_1(data)
│  └─ First-order differencing: x[t] - x[t-1]
│
├─ gen_time_feature(data)
│  └─ Extract: month, day, hour
│
├─ cos_date(month, day, hour)
│  └─ cos(2π × day_of_year / 365)
│
├─ sin_date(month, day, hour)
│  └─ sin(2π × day_of_year / 365)
│
└─ adjust_learning_rate(optimizer, epoch, opt)
   └─ Type4: Exponential decay schedule
```

### 5.2 Evaluation Metrics (metric.py)

```python
metric(name, pred, gt):
├─ MAE = mean(|pred - gt|)
├─ MSE = mean((pred - gt)²)
├─ RMSE = sqrt(MSE)
└─ MAPE = mean(|pred - gt| / gt) × 100%
```

---

## 6. File I/O Architecture

### 6.1 Model Serialization

```
Model ZIP File Structure:
├─ MCANN_encoder.pt       # PyTorch state_dict
├─ MCANN_decoder.pt       # PyTorch state_dict
├─ GMM.pt                 # Sample-wise GMM (pickle)
├─ GM3.pt                 # Point-wise GMM (pickle)
├─ GMM0.pt                # Dataset-wise GMM (pickle)
├─ Norm.txt               # [mean, std] normalization params
└─ opt.txt                # All hyperparameters (key|value format)
```

### 6.2 Data Formats

**Input TSV Format:**
```
datetime                value
1983-07-01 00:00:00    1234.56
1983-07-01 00:30:00    1235.12
...
```

**Output Prediction Format:**
```
timestamp              step  prediction
2019-01-07 03:30:00   0     1234.56
2019-01-07 04:30:00   1     1235.12
...                   71    1250.34
```

---

## 7. Memory Architecture

### 7.1 Memory Footprint

**Training Phase:**
```
Component                    Memory
─────────────────────────────────────
Input Batch (48 × 360 × 8)   ~0.5 MB
Encoder Parameters           ~10 MB
Decoder Parameters           ~10 MB
Hidden States (3 branches)   ~6 MB
Attention Weights            ~2 MB
Gradients                    ~20 MB
Optimizer States             ~40 MB
─────────────────────────────────────
Total (GPU)                  ~90 MB
```

**Inference Phase:**
```
Component                    Memory
─────────────────────────────────────
Model Weights                ~20 MB
Input Sequence (1 × 360 × 8) ~10 KB
Intermediate Activations     ~5 MB
Output (1 × 72)              ~1 KB
─────────────────────────────────────
Total (GPU/CPU)              ~25 MB
```

---

## 8. Computational Complexity

### 8.1 Time Complexity

**Training:**
- **Per Batch**: O(B × T × H²) where B=batch, T=sequence, H=hidden
- **Per Epoch**: O(N/B × B × T × H²) = O(N × T × H²)
- **Total Training**: O(E × N × T × H²) where E=epochs

**Inference:**
- **Single Prediction**: O(T × H²) ≈ O(360 × 512²) ≈ 94M ops
- **Latency**: ~100ms on GPU, ~1s on CPU

### 8.2 Space Complexity

- **Model Parameters**: O(H² + A²) ≈ 512² + 600² ≈ 622K params
- **Training Memory**: O(B × T × H) ≈ 48 × 360 × 512 ≈ 8.8M values
- **Inference Memory**: O(T × H) ≈ 360 × 512 ≈ 184K values

---

## 9. System Integration Points

### 9.1 External Dependencies

```
PyTorch → Neural Network Computation
├─ torch.nn: LSTM, Linear, MultiheadAttention
├─ torch.optim: Adam optimizer
└─ torch.utils.data: DataLoader

NumPy → Numerical Operations
├─ Array manipulation
├─ Statistical functions
└─ Linear algebra

Pandas → Data I/O
├─ Read/Write TSV files
├─ DateTime handling
└─ Data filtering

Scikit-learn → Clustering
├─ GaussianMixture
├─ Probability estimation
└─ Component ordering

Matplotlib → Visualization
└─ Plot predictions vs ground truth
```

### 9.2 API Interfaces

**Training API:**
```python
# Command-line
python run.py --arg_file models/Stevens_Creek.txt

# Programmatic
from run import Options
from data_provider.DS import DS
from models.Group_GMM5 import DAN

opt = Options().parse()
ds = DS(opt, trainX)
model = DAN(opt, ds)
model.train()
```

**Inference API:**
```python
# Command-line
python predict.py --model_path "model.zip" --test_time "2019-01-07"

# Programmatic
from run import Options
opt_manager = Options()
model = opt_manager.get_model("model.zip")
predictions, gt = model.test_single("2019-01-07 03:30:00")
```

---

## 10. Scalability Considerations

### 10.1 Horizontal Scaling
- **Multi-GPU Training**: Use `torch.nn.DataParallel`
- **Distributed Training**: Use `torch.distributed`
- **Batch Processing**: Increase batch size proportionally

### 10.2 Vertical Scaling
- **Memory Optimization**: Gradient checkpointing
- **Computation Optimization**: Mixed precision training (FP16)
- **Storage Optimization**: Model quantization (INT8)

### 10.3 Production Deployment
- **Model Export**: ONNX format for cross-platform deployment
- **Inference Server**: TorchServe or TensorRT
- **API Gateway**: REST API with FastAPI/Flask
- **Monitoring**: MLflow or TensorBoard for tracking

---

## Summary

The MC-ANN system is a **hierarchical, attention-based architecture** that combines:
1. **Multi-level GMM clustering** for data regime identification
2. **Parallel LSTM branches** for specialized pattern learning
3. **Attention mechanisms** for dynamic weighting
4. **Differencing & normalization** for stability
5. **Oversampling strategies** for extreme event focus

The architecture is **modular**, **scalable**, and designed for **real-time forecasting** of time series data with extreme events.
