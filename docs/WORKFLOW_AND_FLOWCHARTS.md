# MC-ANN Workflow and Flowcharts

## Table of Contents
1. [High-Level Workflow](#1-high-level-workflow)
2. [Detailed Training Workflow](#2-detailed-training-workflow)
3. [Detailed Inference Workflow](#3-detailed-inference-workflow)
4. [Data Processing Workflow](#4-data-processing-workflow)
5. [GMM Clustering Workflow](#5-gmm-clustering-workflow)
6. [Model Forward Pass](#6-model-forward-pass)
7. [System State Diagram](#7-system-state-diagram)

---

## 1. High-Level Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    START: MC-ANN System                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
         ┌───────────────────────────────┐
         │   Load Configuration          │
         │   • Hyperparameters          │
         │   • Dataset paths             │
         │   • Training settings         │
         └───────────────┬───────────────┘
                         │
                         ↓
         ┌───────────────────────────────┐
         │   Load & Preprocess Data      │
         │   • Read TSV files            │
         │   • Handle missing values     │
         │   • Normalize data            │
         └───────────────┬───────────────┘
                         │
                         ↓
         ┌───────────────────────────────┐
         │   Build GMM Models            │
         │   • Point-wise GMM (GM3)      │
         │   • Dataset-wise GMM (GMM0)   │
         │   • Sample-wise GMM (GMM)     │
         └───────────────┬───────────────┘
                         │
                         ↓
         ┌───────────────────────────────┐
         │   Generate Features           │
         │   • Extreme scores            │
         │   • GMM probabilities         │
         │   • Time features             │
         └───────────────┬───────────────┘
                         │
                         ↓
         ┌───────────────────────────────┐
         │   Create Data Splits          │
         │   • Training set              │
         │   • Validation set            │
         │   • Test set                  │
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ↓                               ↓
    ┌─────────┐                   ┌─────────────┐
    │ TRAIN   │                   │ INFERENCE   │
    │ MODE    │                   │ MODE        │
    └────┬────┘                   └──────┬──────┘
         │                               │
         ↓                               ↓
    ┌─────────────────┐          ┌──────────────┐
    │ Train Model     │          │ Load Model   │
    │ • Forward pass  │          │ • encoder.pt │
    │ • Compute loss  │          │ • decoder.pt │
    │ • Backprop      │          │ • GMM models │
    │ • Validate      │          └──────┬───────┘
    └────┬────────────┘                 │
         │                               │
         ↓                               ↓
    ┌─────────────────┐          ┌──────────────┐
    │ Save Model      │          │ Predict      │
    │ • Best weights  │          │ • 3 days     │
    │ • GMM models    │          │ • Hourly     │
    │ • Parameters    │          └──────┬───────┘
    └────┬────────────┘                 │
         │                               │
         └───────────────┬───────────────┘
                         ↓
         ┌───────────────────────────────┐
         │   Evaluate Results            │
         │   • Compute RMSE              │
         │   • Compute MAPE              │
         │   • Visualize predictions     │
         └───────────────┬───────────────┘
                         │
                         ↓
         ┌───────────────────────────────┐
         │   Save Outputs                │
         │   • Predictions TSV           │
         │   • Plots PNG                 │
         │   • Metrics LOG               │
         └───────────────┬───────────────┘
                         │
                         ↓
                    ┌─────────┐
                    │   END   │
                    └─────────┘
```

---

## 2. Detailed Training Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                             │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
            ┌────────────────────────┐
            │ Initialize Model       │
            │ • EncoderLSTM         │
            │ • DecoderLSTM         │
            │ • Optimizers          │
            │ • Loss Functions      │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Load Training Data     │
            │ • Batch size: 48       │
            │ • Samples: 20K-40K     │
            │ • Oversampling: 30-40% │
            └────────┬───────────────┘
                     │
                     ↓
        ╔════════════════════════════════╗
        ║   FOR EACH EPOCH (max 50)     ║
        ╚════════════════════════════════╝
                     │
                     ↓
        ┌────────────────────────────────┐
        │ Training Loop                  │
        │ ┌────────────────────────────┐│
        │ │ FOR EACH BATCH:           ││
        │ │                            ││
        │ │ 1. Get batch data         ││
        │ │    x: [48, 360, 8]        ││
        │ │    y: [48, 72, 5]         ││
        │ │                            ││
        │ │ 2. Forward pass           ││
        │ │    h,c,w = Encoder(x)     ││
        │ │    pred = Decoder(y,h,c,w)││
        │ │                            ││
        │ │ 3. Denormalize            ││
        │ │    pred = pred*std + mean ││
        │ │    pred = pred + y_prev   ││
        │ │                            ││
        │ │ 4. Compute loss           ││
        │ │    loss = MSE(pred, y_gt) ││
        │ │                            ││
        │ │ 5. Backpropagation        ││
        │ │    loss.backward()        ││
        │ │    optimizer.step()       ││
        │ │                            ││
        │ │ 6. Accumulate loss        ││
        │ │    total_loss += loss     ││
        │ └────────────────────────────┘│
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ Validation Loop                │
        │ ┌────────────────────────────┐│
        │ │ FOR EACH VAL POINT (60):  ││
        │ │                            ││
        │ │ 1. Extract input (360pts) ││
        │ │                            ││
        │ │ 2. Predict (72 pts)       ││
        │ │                            ││
        │ │ 3. Compute RMSE           ││
        │ │    rmse = sqrt(MSE)       ││
        │ │                            ││
        │ │ 4. Accumulate             ││
        │ │    total_rmse += rmse     ││
        │ └────────────────────────────┘│
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ Check Validation Performance   │
        │                                 │
        │ IF val_loss < min_loss:        │
        │    Save model weights          │
        │    min_loss = val_loss         │
        │    early_stop_counter = 0      │
        │ ELSE:                           │
        │    early_stop_counter += 1     │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ Adjust Learning Rate           │
        │ • Type4: Exponential decay     │
        │ • LR = LR * decay_factor       │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ Check Early Stopping           │
        │                                 │
        │ IF early_stop_counter >= 4:    │
        │    BREAK training loop         │
        └────────────┬───────────────────┘
                     │
                     ↓
        ╔════════════════════════════════╗
        ║   END EPOCH LOOP               ║
        ╚════════════════════════════════╝
                     │
                     ↓
        ┌────────────────────────────────┐
        │ Save Final Model               │
        │ • Create ZIP file              │
        │ • Include encoder.pt           │
        │ • Include decoder.pt           │
        │ • Include GMM models           │
        │ • Include opt.txt              │
        │ • Include Norm.txt             │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ Training Complete              │
        │ • Display final metrics        │
        │ • Save training logs           │
        └────────────────────────────────┘
```

---

## 3. Detailed Inference Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
            ┌────────────────────────┐
            │ Load Model Package     │
            │ • Unzip model.zip      │
            └────────┬───────────────┘
                     │
           ┌─────────┴─────────┐
           │                   │
           ↓                   ↓
    ┌──────────────┐    ┌──────────────┐
    │ Load Weights │    │ Load Config  │
    │ • encoder.pt │    │ • opt.txt    │
    │ • decoder.pt │    │ • Norm.txt   │
    └──────┬───────┘    └──────┬───────┘
           │                   │
           └─────────┬─────────┘
                     ↓
            ┌────────────────────────┐
            │ Load GMM Models        │
            │ • GM3.pt (point-wise)  │
            │ • GMM0.pt (dataset)    │
            │ • GMM.pt (sample)      │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Load Time Series Data  │
            │ • Read TSV file        │
            │ • Parse timestamps     │
            │ • Extract values       │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Preprocess Data        │
            │ • Normalize (mean,std) │
            │ • Handle NaN values    │
            │ • Compute time features│
            └────────┬───────────────┘
                     │
                     ↓
        ╔════════════════════════════════╗
        ║  FOR EACH TEST POINT          ║
        ╚════════════════════════════════╝
                     │
                     ↓
        ┌────────────────────────────────┐
        │ 1. Extract Input Sequence      │
        │    • Get last 360 timestamps   │
        │    • Extract normalized values │
        │    • Verify no NaN values      │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ 2. Compute Point Features      │
        │    • GM3 probabilities         │
        │    • Extreme scores            │
        │    • GMM0 probabilities        │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ 3. Compute Sequence Features   │
        │    • Extract last 72 values    │
        │    • GMM predict_proba()       │
        │    • Order by weights          │
        │    • Broadcast to 360 steps    │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ 4. Prepare Input Tensor        │
        │    x = [1, 360, 8]             │
        │    • Dim 0: normalized value   │
        │    • Dim 1: extreme score      │
        │    • Dim 2-4: GMM0 probs       │
        │    • Dim 5-7: GMM probs        │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ 5. Prepare Time Features       │
        │    y_time = [1, 72, 2]         │
        │    • Dim 0: cos(date)          │
        │    • Dim 1: sin(date)          │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ 6. Forward Pass                │
        │    ┌─────────────────────────┐ │
        │    │ Encoder:                │ │
        │    │   h, c, w = encoder(x)  │ │
        │    │                          │ │
        │    │ Decoder:                │ │
        │    │   pred = decoder(       │ │
        │    │     y_time, h, c, w)    │ │
        │    └─────────────────────────┘ │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ 7. Denormalize Predictions     │
        │    • pred_denorm = pred*std    │
        │    •             + mean        │
        │    • pred_final = pred_denorm  │
        │    •            + y_prev       │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ 8. Store Predictions           │
        │    • Timestamp                 │
        │    • 72 hourly predictions     │
        │    • Ground truth (if avail)   │
        └────────────┬───────────────────┘
                     │
                     ↓
        ╔════════════════════════════════╗
        ║  END TEST POINT LOOP           ║
        ╚════════════════════════════════╝
                     │
                     ↓
        ┌────────────────────────────────┐
        │ Aggregate Results              │
        │ • Collect all predictions      │
        │ • Compute metrics (RMSE,MAPE)  │
        │ • Generate visualizations      │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │ Save Outputs                   │
        │ • Predictions TSV              │
        │ • Plots PNG                    │
        │ • Metrics TXT                  │
        └────────────────────────────────┘
```

---

## 4. Data Processing Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                 DATA PROCESSING PIPELINE                      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
            ┌────────────────────────┐
            │ Read Raw TSV File      │
            │ columns: [datetime,    │
            │          value]        │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Parse Timestamps       │
            │ • Convert to datetime  │
            │ • Sort chronologically │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Handle Missing Values  │
            │ • Identify NaN         │
            │ • Tag as invalid       │
            │ • Skip in training     │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Extract Date Range     │
            │ • start_point          │
            │ • train_point          │
            │ • test_start           │
            │ • test_end             │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Compute Differences    │
            │ diff[t] = x[t] - x[t-1]│
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Log Transform          │
            │ x_log = log(x + ε)     │
            │ where ε = 1e-10        │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Standardize            │
            │ x_norm = (x_log - μ)/σ │
            │ • Compute μ (mean)     │
            │ • Compute σ (std)      │
            │ • Save for later       │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Generate Time Features │
            │ • month = datetime.month│
            │ • day = datetime.day   │
            │ • hour = datetime.hour │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Cyclical Encoding      │
            │ • cos_date = cos(2π*d/365)│
            │ • sin_date = sin(2π*d/365)│
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Build GM3 Model        │
            │ • Fit on all data      │
            │ • 3 components         │
            │ • Get extreme scores   │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Build GMM0 Model       │
            │ • Sample 200K windows  │
            │ • Fit on samples       │
            │ • Order components     │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Compute Point Features │
            │ • GM3 probabilities    │
            │ • GMM0 probabilities   │
            │ • Add to feature tensor│
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Generate Training Tags │
            │ • 0: Invalid (NaN)     │
            │ • 1: Available         │
            │ • 2: Validation        │
            │ • 3: Near validation   │
            │ • 4: In training       │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Sample Training Data   │
            │ • Random sampling      │
            │ • Oversampling extreme │
            │ • Avoid val points     │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Build GMM Model        │
            │ • Fit on train samples │
            │ • Compute sample probs │
            │ • Add to features      │
            └────────┬───────────────┘
                     │
                     ↓
            ┌────────────────────────┐
            │ Create DataLoader      │
            │ • Batch size: 48       │
            │ • Shuffle: True        │
            │ • Num workers: 2       │
            └────────────────────────┘
```

---

## 5. GMM Clustering Workflow

```
┌──────────────────────────────────────────────────────────────┐
│              THREE-LEVEL GMM HIERARCHY                        │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ LEVEL 1: GM3 (Point-wise Clustering)                         │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT: All non-NaN normalized values [N × 1]                │
│         ↓                                                     │
│  ┌────────────────────────────────────┐                      │
│  │ GaussianMixture(n_components=3)    │                      │
│  │ • Initialize with kmeans           │                      │
│  │ • EM algorithm to convergence      │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Learn 3 Gaussian Components:       │                      │
│  │ • μ₀, σ₀ (low values)             │                      │
│  │ • μ₁, σ₁ (medium values)          │                      │
│  │ • μ₂, σ₂ (high values)            │                      │
│  │ • π₀, π₁, π₂ (weights)            │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Predict Probabilities:             │                      │
│  │ P(component|x) for each x          │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Compute Extreme Score:             │                      │
│  │ score = 1 - Σ P(k|x) × πₖ        │                      │
│  │       = P(outlier)                 │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  OUTPUT: Extreme scores [N × 1]                              │
│          • High score → extreme value                         │
│          • Low score → typical value                          │
│                                                                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ LEVEL 2: GMM0 (Dataset-wise Clustering)                      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT: Random 200K samples of last 72 values [200K × 1]    │
│         ↓                                                     │
│  ┌────────────────────────────────────┐                      │
│  │ Random Sampling:                   │                      │
│  │ FOR i = 1 to 200K:                │                      │
│  │   idx = random(0, N-72)           │                      │
│  │   IF no NaN in [idx:idx+72]:      │                      │
│  │     sample[i] = x[idx]            │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ GaussianMixture(n_components=3)    │                      │
│  │ • Fit on sampled data              │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Order Components by Weight:        │                      │
│  │ • order1 = argmax(weights)         │                      │
│  │ • order2 = argmin(weights)         │                      │
│  │ • order3 = remaining               │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Predict for All Data:              │                      │
│  │ probs = predict_proba(x_all)       │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Reorder Probabilities:             │                      │
│  │ [P(high), P(low), P(mid)]          │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  OUTPUT: Ordered probabilities [N × 3]                       │
│          • Dim 0: P(most frequent)                           │
│          • Dim 1: P(least frequent)                          │
│          • Dim 2: P(medium frequency)                        │
│                                                                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ LEVEL 3: GMM (Sample-wise Clustering)                        │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT: Last 72 extreme scores from train samples [M × 72]  │
│         ↓                                                     │
│  ┌────────────────────────────────────┐                      │
│  │ Extract Sequences:                 │                      │
│  │ FOR each training sample:          │                      │
│  │   seq = extreme_scores[-72:]       │                      │
│  │   flatten to [1 × 72]              │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ GaussianMixture(n_components=3)    │                      │
│  │ • Fit on flattened sequences       │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Order Components by Weight:        │                      │
│  │ • order1 = argmin(weights)         │                      │
│  │ • order2 = argmax(weights)         │                      │
│  │ • order3 = remaining               │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Predict for All Samples:           │                      │
│  │ probs = predict_proba(seqs)        │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Reorder and Broadcast:             │                      │
│  │ [P0, P1, P2] → repeat 360 times   │                      │
│  │ → [M × 360 × 3]                    │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  OUTPUT: Sequence probabilities [M × 360 × 3]               │
│          • Dim 0: P(low extreme)                             │
│          • Dim 1: P(high extreme)                            │
│          • Dim 2: P(normal)                                  │
│                                                                │
└──────────────────────────────────────────────────────────────┘

                    COMBINED OUTPUT
                         ↓
        ┌────────────────────────────────┐
        │ Final Feature Tensor:          │
        │ [Batch, Sequence, 8 features]  │
        │                                 │
        │ Dim 0: Normalized value        │
        │ Dim 1: Extreme score (GM3)     │
        │ Dim 2-4: Dataset probs (GMM0)  │
        │ Dim 5-7: Sample probs (GMM)    │
        └────────────────────────────────┘
```

---

## 6. Model Forward Pass

```
┌──────────────────────────────────────────────────────────────┐
│                   FORWARD PASS FLOW                           │
└──────────────────────────────────────────────────────────────┘

INPUT: x = [Batch=48, Sequence=360, Features=8]
       y_time = [Batch=48, Horizon=72, TimeFeats=2]

┌──────────────────────────────────────────────────────────────┐
│                      ENCODER PHASE                            │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Extract Input Features:            │                      │
│  │ x0 = x[:, :, 0:2]  # [B,360,2]    │                      │
│  │ gmm_probs = x[:, :, 2:8] # [B,360,6]│                    │
│  └────────────┬───────────────────────┘                      │
│               │                                                │
│               ├─────────────────┬─────────────────┐          │
│               ↓                 ↓                 ↓          │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ LSTM Branch 0    │  │ LSTM Branch 1│  │ LSTM Branch 2│  │
│  │ Input: x0        │  │ Input: x0    │  │ Input: x0    │  │
│  │ [B, 360, 2]      │  │ [B, 360, 2]  │  │ [B, 360, 2]  │  │
│  │      ↓           │  │      ↓       │  │      ↓       │  │
│  │ LSTM(2→512, L)   │  │ LSTM(2→512,L)│  │ LSTM(2→512,L)│  │
│  │      ↓           │  │      ↓       │  │      ↓       │  │
│  │ out: [B,360,512] │  │ out:[B,360,512]│ │ out:[B,360,512]│  │
│  │ h0: [L, B, 512]  │  │ h1:[L, B, 512]│ │ h2:[L, B, 512]│  │
│  │ c0: [L, B, 512]  │  │ c1:[L, B, 512]│ │ c2:[L, B, 512]│  │
│  └──────────────────┘  └──────────────┘  └──────────────┘  │
│                                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Attention Weight Computation:      │                      │
│  │ ww0 = x[:, -72:, 2:5]  # GMM probs│                      │
│  │ ww1 = x[:, -72:, 5:8]  # Sample   │                      │
│  │ ww = ww0 + ww1 * seq_weight        │                      │
│  │      # [B, 72, 3]                  │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Process Each Component:            │                      │
│  │                                     │                      │
│  │ Component 0: ww[:,:,0] [B,72,1]    │                      │
│  │   ↓                                 │                      │
│  │ Linear(1→150)                       │                      │
│  │   ↓                                 │                      │
│  │ Add PositionalEmbedding(150)       │                      │
│  │   ↓                                 │                      │
│  │ MultiHeadAttention(150, 4 heads)   │                      │
│  │   ↓                                 │                      │
│  │ Add & BatchNorm                    │                      │
│  │   ↓                                 │                      │
│  │ MultiHeadAttention(150, 4 heads)   │                      │
│  │   ↓                                 │                      │
│  │ Add & BatchNorm                    │                      │
│  │   ↓                                 │                      │
│  │ Linear(150→1)                       │                      │
│  │   ↓                                 │                      │
│  │ w0 [B, 72, 1]                      │                      │
│  │                                     │                      │
│  │ (Same for Component 1 → w1)        │                      │
│  │ (Same for Component 2 → w2)        │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Concatenate & Softmax:             │                      │
│  │ ww = concat([w0, w1, w2], dim=2)   │                      │
│  │    = [B, 72, 3]                    │                      │
│  │ ww = softmax(ww, dim=2)            │                      │
│  │    # Σ ww[i,j,:] = 1              │                      │
│  └────────────────────────────────────┘                      │
│                                                                │
│  ENCODER OUTPUT:                                              │
│    • h = [h0, h1, h2]  (3 hidden states)                    │
│    • c = [c0, c1, c2]  (3 cell states)                      │
│    • ww = [B, 72, 3]   (attention weights)                  │
│                                                                │
└──────────────────────────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                      DECODER PHASE                            │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT:                                                       │
│    • y_time: [B, 72, 2]  (cos/sin date)                     │
│    • h: [h0, h1, h2]                                         │
│    • c: [c0, c1, c2]                                         │
│    • ww: [B, 72, 3]                                          │
│                                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Branch 0:                          │                      │
│  │ LSTM(input=y_time,                 │                      │
│  │      init_h=h0, init_c=c0)         │                      │
│  │   ↓                                 │                      │
│  │ out0: [B, 72, 512]                 │                      │
│  │   ↓                                 │                      │
│  │ Linear(512→1)                       │                      │
│  │   ↓                                 │                      │
│  │ out0: [B, 72]                      │                      │
│  └────────────────────────────────────┘                      │
│                                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Branch 1:                          │                      │
│  │ LSTM(input=y_time,                 │                      │
│  │      init_h=h1, init_c=c1)         │                      │
│  │   ↓                                 │                      │
│  │ out1: [B, 72, 512]                 │                      │
│  │   ↓                                 │                      │
│  │ Linear(512→1)                       │                      │
│  │   ↓                                 │                      │
│  │ out1: [B, 72]                      │                      │
│  └────────────────────────────────────┘                      │
│                                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Branch 2:                          │                      │
│  │ LSTM(input=y_time,                 │                      │
│  │      init_h=h2, init_c=c2)         │                      │
│  │   ↓                                 │                      │
│  │ out2: [B, 72, 512]                 │                      │
│  │   ↓                                 │                      │
│  │ Linear(512→1)                       │                      │
│  │   ↓                                 │                      │
│  │ out2: [B, 72]                      │                      │
│  └────────────────────────────────────┘                      │
│                                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Weighted Combination:              │                      │
│  │                                     │                      │
│  │ w0 = ww[:, :, 0]  # [B, 72]       │                      │
│  │ w1 = ww[:, :, 1]  # [B, 72]       │                      │
│  │ w2 = ww[:, :, 2]  # [B, 72]       │                      │
│  │                                     │                      │
│  │ out = w0 ⊙ out0                    │                      │
│  │     + w1 ⊙ out1                    │                      │
│  │     + w2 ⊙ out2                    │                      │
│  │     # [B, 72]                      │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  DECODER OUTPUT: [B, 72]                                     │
│    (normalized predictions)                                   │
│                                                                │
└──────────────────────────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                  POST-PROCESSING                              │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Denormalize:                       │                      │
│  │ out = out * std + mean             │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  ┌────────────────────────────────────┐                      │
│  │ Add Previous Value:                │                      │
│  │ out = out + y_prev                 │                      │
│  │     # Reverse differencing         │                      │
│  └────────────┬───────────────────────┘                      │
│               ↓                                                │
│  FINAL OUTPUT: [B, 72]                                       │
│    (denormalized predictions in original scale)              │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. System State Diagram

```
                    ┌─────────────┐
                    │  START/IDLE │
                    └──────┬──────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ↓                             ↓
    ┌────────────────┐          ┌─────────────────┐
    │  TRAIN MODE    │          │ INFERENCE MODE  │
    └────────┬───────┘          └─────────┬───────┘
             │                             │
             ↓                             ↓
    ┌────────────────┐          ┌─────────────────┐
    │ LOADING DATA   │          │ LOADING MODEL   │
    └────────┬───────┘          └─────────┬───────┘
             │                             │
             ↓                             │
    ┌────────────────┐                    │
    │ PREPROCESSING  │                    │
    └────────┬───────┘                    │
             │                             │
             ↓                             │
    ┌────────────────┐                    │
    │ BUILDING GMMs  │                    │
    └────────┬───────┘                    │
             │                             │
             ↓                             │
    ┌────────────────┐                    │
    │ GENERATING     │                    │
    │ FEATURES       │                    │
    └────────┬───────┘                    │
             │                             │
             ↓                             │
    ┌────────────────┐                    │
    │ CREATING       │                    │
    │ DATALOADERS    │                    │
    └────────┬───────┘                    │
             │                             │
             ↓                             │
    ┌────────────────┐          ┌─────────────────┐
    │ TRAINING       │          │ LOADING DATA    │
    │ • Forward      │          └─────────┬───────┘
    │ • Backward     │                    │
    │ • Validation   │                    ↓
    └────────┬───────┘          ┌─────────────────┐
             │                   │ PREPROCESSING   │
             │                   │ (with saved     │
             │                   │  params)        │
             │                   └─────────┬───────┘
             │                             │
             │                             ↓
             ↓                   ┌─────────────────┐
    ┌────────────────┐          │ PREDICTING      │
    │ SAVING MODEL   │          │ • For each      │
    │ • encoder.pt   │          │   test point    │
    │ • decoder.pt   │          │ • Forward pass  │
    │ • GMMs         │          │ • Denormalize   │
    │ • params       │          └─────────┬───────┘
    └────────┬───────┘                    │
             │                             │
             └──────────┬──────────────────┘
                        ↓
             ┌─────────────────┐
             │ EVALUATION      │
             │ • Compute RMSE  │
             │ • Compute MAPE  │
             │ • Visualization │
             └────────┬────────┘
                      │
                      ↓
             ┌─────────────────┐
             │ SAVING RESULTS  │
             │ • predictions   │
             │ • plots         │
             │ • metrics       │
             └────────┬────────┘
                      │
                      ↓
             ┌─────────────────┐
             │ COMPLETE/IDLE   │
             └─────────────────┘
```

---

## Summary

This document provides comprehensive **flowcharts and workflows** for the MC-ANN system, covering:

1. **High-level system flow** from data loading to results
2. **Detailed training loop** with validation and early stopping
3. **Inference pipeline** for making predictions
4. **Data processing steps** including normalization and feature generation
5. **Three-level GMM hierarchy** for clustering-based features
6. **Model forward pass** through encoder-decoder architecture
7. **System states** and transitions

These diagrams help understand:
- **Control flow**: How the system progresses through stages
- **Data flow**: How data transforms at each step
- **Dependencies**: What must happen before what
- **Decision points**: Where branching occurs
- **Parallelism**: Where operations happen simultaneously

Use these workflows as a **reference** when:
- Understanding the codebase
- Debugging issues
- Implementing modifications
- Training new models
- Deploying for inference
