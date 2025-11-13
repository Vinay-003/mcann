# MC-ANN Documentation

Complete documentation for the MC-ANN (Mixture Clustering-Based Attention Neural Network) project.

## 📚 Documentation Index

### 1. [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md)
**High-level introduction to MC-ANN**
- What is MC-ANN?
- Key innovations and contributions
- Model architecture overview
- Dataset information
- Applications and use cases
- Citation information

**Read this first** to understand what MC-ANN is and what it does.

---

### 2. [SETUP_AND_RUN_GUIDE.md](./SETUP_AND_RUN_GUIDE.md)
**Complete step-by-step setup and execution instructions**
- Environment setup (Conda, PyTorch)
- Dataset download and installation
- Training models (all 5 reservoirs)
- Running inference
- Validation of paper results
- Troubleshooting common issues
- Performance optimization tips

**Read this** when you want to run MC-ANN on your system.

---

### 3. [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)
**Detailed technical architecture documentation**
- Data layer architecture
- Model layer (Encoder/Decoder) details
- Three-level GMM hierarchy
- Training pipeline
- Inference pipeline
- Memory and computational complexity
- API interfaces
- System integration points

**Read this** for deep technical understanding of how MC-ANN works internally.

---

### 4. [WORKFLOW_AND_FLOWCHARTS.md](./WORKFLOW_AND_FLOWCHARTS.md)
**Visual workflows and flowcharts**
- High-level workflow
- Detailed training workflow
- Detailed inference workflow
- Data processing pipeline
- GMM clustering workflow
- Model forward pass
- System state diagram

**Read this** for visual understanding of the system flow.

---

### 5. [PAPER_SUMMARY.md](./PAPER_SUMMARY.md)
**Comprehensive analysis of the TPAMI 2025 paper**
- Problem definition and motivation
- Related work comparison
- Detailed methodology
- Experimental setup
- Key results and findings
- Discussion and insights
- Theoretical contributions
- Future research directions

**Read this** to understand the research context and scientific contributions.

---

### 6. [TECHNICAL_GLOSSARY.md](./TECHNICAL_GLOSSARY.md)
**Beginner-friendly explanations of ALL technical terms**
- Machine Learning basics
- Neural Networks concepts
- Time Series terminology
- Model architecture terms
- Training & optimization
- Evaluation metrics
- Data processing
- Mathematical operations
- Programming tools
- MC-ANN specific terms

**Read this** if you're new to ML or need to understand technical jargon.

---

## 🚀 Quick Start

### For First-Time Users (No ML Background):
1. Read [TECHNICAL_GLOSSARY.md](./TECHNICAL_GLOSSARY.md) - Learn the basics (20 min)
2. Read [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) - Understand what MC-ANN is (10 min)
3. Follow [SETUP_AND_RUN_GUIDE.md](./SETUP_AND_RUN_GUIDE.md) - Run the code (30-60 min)
4. Reference glossary whenever you see unfamiliar terms

### For Researchers:
1. Read [PAPER_SUMMARY.md](./PAPER_SUMMARY.md) (30 min)
2. Read [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) (20 min)
3. Reference [TECHNICAL_GLOSSARY.md](./TECHNICAL_GLOSSARY.md) for specific terms
4. Study the code with architecture reference

### For Developers:
1. Read [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) (20 min)
2. Reference [WORKFLOW_AND_FLOWCHARTS.md](./WORKFLOW_AND_FLOWCHARTS.md) for debugging
3. Use [TECHNICAL_GLOSSARY.md](./TECHNICAL_GLOSSARY.md) as reference
4. Modify code as needed

---

## 📖 Documentation Structure

```
docs/
├── README.md                          # This file - Documentation index
├── PROJECT_OVERVIEW.md                # High-level project introduction
├── SETUP_AND_RUN_GUIDE.md            # Installation and execution guide
├── SYSTEM_ARCHITECTURE.md            # Technical architecture details
├── WORKFLOW_AND_FLOWCHARTS.md        # Visual workflows and diagrams
├── PAPER_SUMMARY.md                  # Research paper analysis
├── TECHNICAL_GLOSSARY.md             # Beginner-friendly term explanations
├── QUICK_REFERENCE.md                # Quick command reference
└── MC-ANN_Paper.pdf                  # Original TPAMI 2025 paper
```

---

## 🎯 Documentation by Task

### Task: **Train a Model**
1. [SETUP_AND_RUN_GUIDE.md](./SETUP_AND_RUN_GUIDE.md) → Step 3
2. Command: `python run.py --arg_file models/Stevens_Creek.txt`

### Task: **Run Inference**
1. [SETUP_AND_RUN_GUIDE.md](./SETUP_AND_RUN_GUIDE.md) → Step 4
2. Command: `python predict.py --model_path "output/model.zip" --test_time "2019-01-07"`

### Task: **Understand the Model**
1. [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) → Section 2 (Model Layer)
2. [WORKFLOW_AND_FLOWCHARTS.md](./WORKFLOW_AND_FLOWCHARTS.md) → Section 6 (Forward Pass)

### Task: **Reproduce Paper Results**
1. [SETUP_AND_RUN_GUIDE.md](./SETUP_AND_RUN_GUIDE.md) → Step 6
2. [PAPER_SUMMARY.md](./PAPER_SUMMARY.md) → Section 5 (Key Results)

### Task: **Modify the Architecture**
1. [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) → Section 2 (Model Layer)
2. Edit `models/GMM_Model5.py` (Encoder/Decoder)
3. Edit `models/Group_GMM5.py` (Training logic)

### Task: **Add New Dataset**
1. [WORKFLOW_AND_FLOWCHARTS.md](./WORKFLOW_AND_FLOWCHARTS.md) → Section 4 (Data Processing)
2. Edit `data_provider/DS.py`
3. Create new parameter file in `models/`

### Task: **Troubleshoot Errors**
1. [SETUP_AND_RUN_GUIDE.md](./SETUP_AND_RUN_GUIDE.md) → Step 7 (Common Issues)
2. Check system logs in `output/*/train/model.log`

---

## 🔍 Key Concepts Explained

### What is MC-ANN?
A deep learning model that uses **mixture clustering** (GMM) to identify different data patterns and **attention mechanisms** to combine predictions from specialized neural networks. See [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md).

### What are the 3 GMM Levels?
1. **GM3:** Point-wise clustering (identifies individual extremes)
2. **GMM0:** Dataset-wise clustering (captures global patterns)
3. **GMM:** Sample-wise clustering (classifies entire sequences)

See [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) → Section 2.4.

### How does the Encoder-Decoder work?
- **Encoder:** 3 parallel LSTMs process input → produce hidden states + attention weights
- **Decoder:** 3 parallel LSTMs generate predictions → weighted combination
- **Attention:** Dynamically weights each LSTM branch based on input characteristics

See [WORKFLOW_AND_FLOWCHARTS.md](./WORKFLOW_AND_FLOWCHARTS.md) → Section 6.

### What is Oversampling?
Increasing the representation of extreme events (floods/droughts) in the training set from ~5% to 30-40%. Critical for learning rare but important patterns. See [PAPER_SUMMARY.md](./PAPER_SUMMARY.md) → Section 3.6.

---

## 📊 Performance Summary

| Metric | Baseline (LSTM) | MC-ANN | Improvement |
|--------|----------------|---------|-------------|
| RMSE | 65.4 | **47.8** | **27%** |
| MAPE | 5.8% | **3.9%** | **33%** |
| Extreme MAPE | 12.3% | **6.2%** | **50%** |

See [PAPER_SUMMARY.md](./PAPER_SUMMARY.md) → Section 5 for detailed results.

---

## 🛠️ Technical Specifications

- **Python:** 3.8.8
- **PyTorch:** 1.11.0
- **Input Length:** 360 timesteps (15 days)
- **Output Length:** 72 timesteps (3 days)
- **Hidden Dim:** 512
- **Attention Dim:** 300-600
- **Training Time:** ~5 min/epoch on GPU
- **Inference Time:** ~15ms per prediction

See [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) → Technical Specifications.

---

## 📝 Citation

If you use MC-ANN in your research, please cite:

```bibtex
@ARTICLE{li2025mcann,
  author={Yanhong Li and David Anastasiu}, 
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MC-ANN: A Mixture Clustering-Based Attention Neural Network for Time Series Forecasting}, 
  year={2025}, 
  doi={10.1109/TPAMI.2025.3565224}
}
```

---

## 🤝 Contributing

To contribute documentation:
1. Follow the existing format (Markdown with clear sections)
2. Use diagrams and flowcharts for complex concepts
3. Provide practical examples and code snippets
4. Keep explanations clear and concise

---

## 📧 Support

For questions or issues:
- **Technical Issues:** Check [SETUP_AND_RUN_GUIDE.md](./SETUP_AND_RUN_GUIDE.md) → Step 7
- **Research Questions:** Read [PAPER_SUMMARY.md](./PAPER_SUMMARY.md)
- **Architecture Questions:** Reference [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)

---

## 📅 Document History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-14 | 1.0 | Initial comprehensive documentation created |

---

## 🎓 Learning Path

### Beginner (No ML Background)
1. **[TECHNICAL_GLOSSARY.md](./TECHNICAL_GLOSSARY.md)** - Learn ML basics with simple analogies
2. [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) - What is MC-ANN?
3. [SETUP_AND_RUN_GUIDE.md](./SETUP_AND_RUN_GUIDE.md) - How to run it?
4. Experiment with example notebook
5. [WORKFLOW_AND_FLOWCHARTS.md](./WORKFLOW_AND_FLOWCHARTS.md) - Visual understanding
6. Keep glossary open as reference

### Intermediate (Some ML Knowledge)
1. [PAPER_SUMMARY.md](./PAPER_SUMMARY.md) - Research context
2. [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) - Technical details
3. [WORKFLOW_AND_FLOWCHARTS.md](./WORKFLOW_AND_FLOWCHARTS.md) - System flow
4. [TECHNICAL_GLOSSARY.md](./TECHNICAL_GLOSSARY.md) - Clarify specific terms
5. Modify hyperparameters and retrain

### Advanced (ML Researcher/Engineer)
1. [PAPER_SUMMARY.md](./PAPER_SUMMARY.md) - Full paper analysis
2. [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) - Implementation details
3. [TECHNICAL_GLOSSARY.md](./TECHNICAL_GLOSSARY.md) - Quick reference
4. Study source code with architecture reference
5. Extend to new domains or improve architecture

---

## 🏆 Best Practices

### For Training:
- Use GPU if available (5x faster than CPU)
- Start with pre-configured parameters in `models/*.txt`
- Monitor validation RMSE for early stopping
- Save multiple checkpoints during training

### For Inference:
- Verify input data has no NaN in last 360 timesteps
- Use saved normalization parameters (mean, std)
- Batch multiple predictions for efficiency
- Validate outputs are in reasonable range

### For Development:
- Keep documentation updated with code changes
- Add unit tests for new features
- Use version control (git) for experiments
- Document hyperparameter choices

---

**Happy Forecasting with MC-ANN! 🚀**
