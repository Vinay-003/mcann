# MC-ANN: Mixture Clustering-Based Attention Neural Network for Time Series Forecasting

## Project Overview

**MC-ANN** (Mixture Clustering-Based Attention Neural Network) is a deep learning model for time series forecasting, specifically designed for reservoir water level prediction. The model has been published in **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 2025**.

### Authors
- Yanhong Li
- David Anastasiu

### Citation
```bibtex
@ARTICLE{li2025mcann,
  author={Yanhong Li and David Anastasiu}, 
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MC-ANN: A Mixture Clustering-Based Attention Neural Network for Time Series Forecasting}, 
  year={2025}, 
  doi={10.1109/TPAMI.2025.3565224}
}
```

## Problem Statement

The project addresses the challenge of **accurate time series forecasting for reservoir water levels**, which is critical for:
- Water resource management
- Flood prediction and prevention
- Agricultural planning
- Hydroelectric power generation
- Environmental monitoring

The main challenges in reservoir level forecasting include:
1. **Extreme Events**: Capturing sudden changes (floods, droughts)
2. **Long-term Dependencies**: Modeling patterns across days, weeks, and seasons
3. **Data Irregularities**: Handling missing data and outliers
4. **Multi-scale Patterns**: Different temporal patterns at various scales

## Key Innovation

MC-ANN introduces a **mixture clustering mechanism** combined with **attention-based neural networks** to:

1. **Gaussian Mixture Models (GMM)**: Automatically identify different data regimes (low, normal, high water levels)
2. **Multi-Component Architecture**: Use separate LSTM branches for different data clusters
3. **Attention Mechanism**: Dynamically weight different LSTM branches based on input patterns
4. **Oversampling Strategy**: Focus training on extreme events (floods/droughts)

## Model Architecture

### Core Components

1. **Encoder (EncoderLSTM)**
   - 3 parallel LSTM branches for different data clusters
   - Multi-head attention layers for temporal pattern extraction
   - Positional embeddings for time-awareness
   - Processes input sequence of length 360 (15 days × 24 hours)

2. **Decoder (DecoderLSTM)**
   - 3 parallel LSTM branches (one per cluster)
   - Weighted combination based on attention scores
   - Generates predictions for 72 timesteps (3 days)

3. **Clustering Module**
   - **GM3**: Point-wise GMM (3 components) for extreme value detection
   - **GMM0**: Dataset-wise GMM (3 components) for regime identification
   - **GMM**: Sample-wise GMM (3 components) for sequence classification

### Input Features (8 dimensions)
- **Dim 0**: Normalized water level value
- **Dim 1**: Extreme score (from GM3)
- **Dim 2-4**: Point-wise GMM probabilities (from GMM0)
- **Dim 5-7**: Sample-wise GMM probabilities (from GMM)

### Output
- 72-step ahead predictions (3 days, hourly resolution)

## Dataset

### Source
- **5 reservoir sensors** from the Santa Clara Valley Water District, California
- **Time range**: 1983-2019 (36 years)
- **Resolution**: 30-minute intervals
- **Download**: [MCANN-datasets.zip](https://clp.engr.scu.edu/static/datasets/MCANN-datasets.zip)

### Reservoirs
1. **Almaden** (reservoir_stor_4001_sof24.tsv)
2. **Coyote** (reservoir_stor_4005_sof24.tsv)
3. **Lexington** (reservoir_stor_4007_sof24.tsv)
4. **Stevens Creek** (reservoir_stor_4009_sof24.tsv)
5. **Vasona** (reservoir_stor_4011_sof24.tsv)

### Data Split
- **Training**: 1983-07-01 to 2018-06-30
- **Validation**: 60 random samples from training period
- **Testing**: 2018-07-01 to 2019-07-01 (1 year)

## Technical Specifications

### Environment Requirements
- **Python**: 3.8.8
- **PyTorch**: 1.11.0
- **CUDA**: Compatible GPU (optional, but recommended)
- **OS**: Linux, macOS, or Windows with Anaconda

### Key Hyperparameters
- **Input length**: 360 timesteps (15 days)
- **Output length**: 72 timesteps (3 days)
- **Hidden dimensions**: 512 (LSTM layers)
- **Attention dimensions**: 300-600 (attention layers)
- **Batch size**: 48
- **Learning rate**: 0.001
- **Training samples**: 20,000-40,000
- **Oversampling ratio**: 30-40% extreme events

### Training Strategy
1. **Data Augmentation**: Oversampling of extreme events
2. **Early Stopping**: Validation-based (4 epochs patience)
3. **Learning Rate Adjustment**: Type4 schedule
4. **Normalization**: Log-standard normalization with differencing
5. **Loss Function**: MSE (training), RMSE & MAPE (evaluation)

## Performance

The model achieves state-of-the-art performance on reservoir water level forecasting:
- **RMSE**: Typically < 50 units for normalized predictions
- **MAPE**: Typically < 5% for 3-day ahead forecasts
- **Inference Time**: Real-time capable (~seconds per forecast)

### Advantages
- Better extreme event prediction than baseline models
- Robust to missing data
- Interpretable cluster assignments
- Scalable to multiple sensors

## Applications

1. **Water Resource Management**: Optimize reservoir operations
2. **Flood Forecasting**: Early warning systems
3. **Agricultural Planning**: Irrigation scheduling
4. **Hydroelectric**: Power generation optimization
5. **Environmental**: Ecosystem health monitoring

## Project Structure

```
mcann/
├── data_provider/
│   ├── DS.py                    # Dataset management and preprocessing
│   └── datasets/                # TSV data files (download separately)
├── models/
│   ├── GMM_Model5.py           # Encoder/Decoder LSTM architectures
│   ├── Group_GMM5.py           # Training and inference logic (DAN class)
│   ├── Inference.py            # Standalone inference module
│   └── *.txt                   # Pre-configured parameter files
├── utils/
│   ├── metric.py               # Evaluation metrics (RMSE, MAE, MAPE)
│   └── utils2.py               # Helper functions (normalization, time features)
├── run.py                      # Main training script
├── predict.py                  # Single-point prediction script
├── test.py                     # Testing and visualization script
├── example.ipynb               # Tutorial notebook
├── experiments.ipynb           # Experimental notebook
├── requirements.txt            # Python dependencies
└── docs/                       # Documentation (this folder)
```

## License

This project is open source. Please check the LICENSE file for details.

## Contact & Support

For questions, issues, or collaborations:
- **Paper**: IEEE TPAMI 2025
- **Dataset**: [https://clp.engr.scu.edu/static/datasets/MCANN-datasets.zip](https://clp.engr.scu.edu/static/datasets/MCANN-datasets.zip)
- **Repository Owner**: davidanastasiu

## Acknowledgments

This work uses data from the Santa Clara Valley Water District and was supported by research grants for hydrological forecasting and water resource management.
