# MC-ANN Technical Glossary

**A Beginner-Friendly Guide to All Technical Terms**

This document explains every technical term used in MC-ANN, the paper, and documentation in simple, easy-to-understand language with real-world analogies.

---

## Table of Contents

1. [Machine Learning Basics](#1-machine-learning-basics)
2. [Neural Networks](#2-neural-networks)
3. [Time Series Concepts](#3-time-series-concepts)
4. [Model Architecture](#4-model-architecture)
5. [Training & Optimization](#5-training--optimization)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Data Processing](#7-data-processing)
8. [Mathematical Operations](#8-mathematical-operations)
9. [Programming & Tools](#9-programming--tools)
10. [MC-ANN Specific Terms](#10-mc-ann-specific-terms)

---

## 1. Machine Learning Basics

### Machine Learning (ML)
**Simple:** Teaching computers to learn from examples instead of programming explicit rules.

**Analogy:** Like teaching a child to recognize cats by showing them many cat pictures, rather than explaining "cats have fur, whiskers, four legs..."

**Example:** Instead of writing rules like "if water level > 1000, predict flood," ML learns patterns from historical data.

---

### Deep Learning
**Simple:** A type of machine learning using artificial "brain-like" networks with many layers.

**Analogy:** Like learning to drive - first you learn steering (layer 1), then acceleration (layer 2), then combining them (layer 3), building complex skills from simple ones.

**In MC-ANN:** The model has multiple layers that progressively learn simple patterns → complex patterns → predictions.

---

### Supervised Learning
**Simple:** Learning from labeled examples (input + correct answer).

**Analogy:** Like studying with answer keys - you see the question (water levels for 15 days) and the answer (water level 3 days later).

**In MC-ANN:** We provide historical water levels (input) and what happened later (output), so the model learns to predict.

---

### Training
**Simple:** The process of teaching the model by showing it many examples.

**Analogy:** Like practicing piano - you play scales repeatedly until you get better.

**In MC-ANN:** Showing the model 20,000-40,000 examples of past water levels and what happened next.

**Technical:** The model adjusts its internal parameters to minimize prediction errors.

---

### Inference
**Simple:** Using a trained model to make predictions on new data.

**Analogy:** After learning to drive, actually driving on a new road.

**In MC-ANN:** Taking the last 15 days of water levels and predicting the next 3 days.

**Note:** Training happens once; inference happens many times.

---

### Model
**Simple:** The trained system that makes predictions.

**Analogy:** Like a recipe - once you've perfected it (training), you can cook using it (inference).

**In MC-ANN:** The combination of encoder, decoder, and GMMs that predict water levels.

---

### Parameters / Weights
**Simple:** Numbers inside the model that determine how it makes predictions.

**Analogy:** Like the settings on a camera (ISO, aperture, shutter speed) - adjusting them changes the outcome.

**In MC-ANN:** ~622,000 numbers that the model learns during training.

**Technical:** These are the values in matrices that transform inputs to outputs.

---

### Hyperparameters
**Simple:** Settings you choose before training (not learned by the model).

**Analogy:** Like choosing the oven temperature before baking - you set it, not the cake.

**Examples in MC-ANN:**
- Hidden dimension (512)
- Learning rate (0.001)
- Batch size (48)
- Number of epochs (50)

**Difference from Parameters:** Parameters are learned; hyperparameters are set by humans.

---

### Overfitting
**Simple:** When the model memorizes training data instead of learning general patterns.

**Analogy:** Like memorizing answers to practice tests but failing the real exam because questions are slightly different.

**Example:** Model predicts perfectly on 2018 data but fails on 2019 data.

**In MC-ANN:** Prevented by dropout, early stopping, and validation sets.

---

### Underfitting
**Simple:** When the model is too simple to capture patterns in the data.

**Analogy:** Using a straight line to describe a circular shape - too simple!

**Example:** Using only yesterday's water level to predict tomorrow (ignoring seasonal patterns).

**Solution:** Use more complex models (more layers, more hidden units).

---

### Generalization
**Simple:** The model's ability to perform well on new, unseen data.

**Analogy:** Learning to drive in your neighborhood and being able to drive in another city.

**In MC-ANN:** Trained on 1983-2018 data, tested on 2019 data.

**Goal:** Good generalization = good performance on test data.

---

## 2. Neural Networks

### Neural Network
**Simple:** A computing system inspired by the human brain, made of connected "neurons."

**Analogy:** Like a team where each person (neuron) does a simple task, but together they solve complex problems.

**Structure:**
```
Input → Hidden Layers → Output
  ↓          ↓            ↓
 360      [512 units]    72
```

**In MC-ANN:** Processes 15 days of data through multiple layers to predict 3 days ahead.

---

### Neuron / Unit
**Simple:** A single processing element that takes inputs, performs calculation, outputs result.

**Analogy:** Like a single worker in a factory assembly line.

**Mathematically:** 
```
output = activation(weight1×input1 + weight2×input2 + ... + bias)
```

**In MC-ANN:** 512 neurons in each hidden layer.

---

### Layer
**Simple:** A group of neurons that process data together.

**Analogy:** Like a floor in a building - data flows from ground floor → 2nd floor → 3rd floor.

**Types:**
- **Input Layer:** Receives raw data (360 timesteps × 8 features)
- **Hidden Layers:** Process and transform data (512 units)
- **Output Layer:** Produces final prediction (72 values)

**In MC-ANN:** Multiple LSTM layers stack on top of each other.

---

### Activation Function
**Simple:** A mathematical function that decides if a neuron should "fire" (activate).

**Analogy:** Like a light switch - but instead of just on/off, it can be dimmed.

**Common Types:**

**1. ReLU (Rectified Linear Unit):**
```
If input > 0: output = input
If input ≤ 0: output = 0
```
Like a floor - can't go below zero.

**2. Tanh (Hyperbolic Tangent):**
```
Output ranges from -1 to +1
```
Like a volume knob with max limits.

**3. Sigmoid:**
```
Output ranges from 0 to 1
```
Like a probability (0% to 100%).

**4. Softmax:**
```
Converts numbers to probabilities that sum to 1
```
Like dividing a pie - all slices must add to 100%.

**In MC-ANN:** Uses Tanh in LSTMs, Softmax for attention weights.

---

### Feedforward
**Simple:** Data flowing in one direction: input → hidden → output.

**Analogy:** Like a one-way street - traffic only goes forward.

**Example:**
```
Water levels → LSTM → Predictions
```

**Contrast:** Recurrent networks have feedback loops (see below).

---

### Recurrent Neural Network (RNN)
**Simple:** A neural network with loops - it remembers previous inputs.

**Analogy:** Like reading a book - you remember previous chapters to understand the current one.

**Why for Time Series:** Each timestep depends on previous timesteps.

**Problem:** Simple RNNs forget long-ago information (vanishing gradients).

**Solution:** LSTM (see below).

---

### LSTM (Long Short-Term Memory)
**Simple:** An advanced RNN that can remember information for long periods.

**Analogy:** Like having a notebook - you can write down important facts and reference them later.

**Components:**
1. **Cell State:** Long-term memory (the notebook)
2. **Hidden State:** Short-term memory (what you're thinking now)
3. **Gates:** Decide what to remember, forget, or output

**In MC-ANN:** Core building block - 6 LSTM layers (3 in encoder, 3 in decoder).

**Why Better than RNN:** Solves the "forgetting" problem using gates.

---

### LSTM Gates

**Simple:** Mechanisms that control information flow in LSTMs.

#### 1. Forget Gate
**What:** Decides what to forget from memory.

**Analogy:** Like deleting old emails - keep important ones, delete spam.

**Example:** Forget water levels from summer when predicting winter.

#### 2. Input Gate
**What:** Decides what new information to store.

**Analogy:** Like highlighting important parts of a textbook.

**Example:** Remember recent rainfall data.

#### 3. Output Gate
**What:** Decides what to output based on memory.

**Analogy:** Like choosing which facts from your notes to include in an essay.

**Example:** Use relevant historical patterns for prediction.

---

### Hidden State (h)
**Simple:** The "working memory" of LSTM - information passed between timesteps.

**Analogy:** Like your current thoughts while reading - you remember the last few sentences.

**In MC-ANN:** 
- Shape: [layers, batch, 512]
- Carries information from encoder to decoder

---

### Cell State (c)
**Simple:** The "long-term memory" of LSTM - information retained over many timesteps.

**Analogy:** Like facts you learned years ago and still remember.

**In MC-ANN:**
- Shape: [layers, batch, 512]
- Maintains important patterns (e.g., seasonal trends)

---

### Hidden Dimension / Hidden Size
**Simple:** The number of neurons in a hidden layer.

**Analogy:** Like the width of a highway - more lanes = more capacity.

**In MC-ANN:** 512 units
- **Larger:** More capacity to learn complex patterns, but slower and may overfit
- **Smaller:** Faster but may miss subtle patterns

---

### Bidirectional
**Simple:** Processing data in both forward and backward directions.

**Analogy:** Reading a sentence left-to-right AND right-to-left to understand it better.

**Example:** In "The cat sat on the mat," understanding "sat" uses both "cat" (before) and "on" (after).

**In MC-ANN:** NOT used (uni-directional only) because we can't look into the future.

---

### Encoder
**Simple:** The part that compresses input into a compact representation.

**Analogy:** Like summarizing a book into cliff notes.

**In MC-ANN:**
- Input: 360 timesteps of water levels
- Output: 3 hidden states + 3 cell states + attention weights
- **Role:** Extract important patterns from historical data

---

### Decoder
**Simple:** The part that expands the compressed representation into predictions.

**Analogy:** Like using cliff notes to reconstruct the story.

**In MC-ANN:**
- Input: Hidden/cell states from encoder + time features
- Output: 72 predictions
- **Role:** Generate future water levels

---

### Encoder-Decoder Architecture
**Simple:** Two-stage system: compress input → generate output.

**Analogy:** Like translating: understand English sentence (encoder) → produce Spanish sentence (decoder).

**In MC-ANN:**
```
15 days of data → Encoder → compact representation → Decoder → 3 days prediction
```

**Benefit:** Separates understanding from generation.

---

### Seq2Seq (Sequence-to-Sequence)
**Simple:** Model that converts one sequence to another sequence.

**Analogy:** Like machine translation: English sentence → French sentence.

**In MC-ANN:**
```
Sequence of past 360 hours → Sequence of future 72 hours
```

**Other Examples:**
- Speech recognition: audio sequence → text sequence
- Video captioning: video frames → text description

---

## 3. Time Series Concepts

### Time Series
**Simple:** Data points collected over time at regular intervals.

**Analogy:** Like keeping a daily diary of temperature readings.

**Examples:**
- Stock prices (every second)
- Weather data (every hour)
- Water levels (every 30 minutes)

**In MC-ANN:** 36 years of reservoir water level measurements.

---

### Timestep
**Simple:** A single point in time in your time series.

**Analogy:** Like one entry in your diary.

**In MC-ANN:** 
- One timestep = one hour
- 360 timesteps = 15 days (input)
- 72 timesteps = 3 days (output)

---

### Sequence
**Simple:** A series of consecutive timesteps.

**Analogy:** Like a chapter in your diary (multiple days).

**In MC-ANN:**
- Input sequence: 360 hours
- Output sequence: 72 hours

---

### Horizon
**Simple:** How far into the future you're predicting.

**Analogy:** Like weather forecast - 3-day forecast vs. 10-day forecast.

**In MC-ANN:** 72 hours (3 days) ahead.

**Trade-off:** 
- Short horizon (1 day): More accurate
- Long horizon (7 days): Less accurate but more useful

---

### Lag / Window
**Simple:** The amount of historical data used for prediction.

**Analogy:** How far back you look before making a decision.

**In MC-ANN:** 360 hours (15 days) of history.

**Choosing Window Size:**
- Too small: Miss long-term patterns
- Too large: Slow and may include irrelevant data

---

### Forecasting
**Simple:** Predicting future values based on past values.

**Analogy:** Like predicting tomorrow's weather based on today's conditions.

**Types:**
- **One-step:** Predict only next value
- **Multi-step:** Predict multiple future values (MC-ANN does this)

---

### Univariate vs Multivariate

**Univariate Time Series:**
**Simple:** Only one variable tracked over time.
**Example:** Just water level (single number per timestep).

**Multivariate Time Series:**
**Simple:** Multiple variables tracked over time.
**Example:** Water level + rainfall + temperature (multiple numbers per timestep).

**In MC-ANN:** Primarily univariate (water level), but uses additional features.

---

### Seasonality
**Simple:** Regular patterns that repeat over time (daily, weekly, yearly).

**Analogy:** Like how it's hot every summer and cold every winter.

**Examples:**
- Water levels higher in rainy season
- Traffic heavier during rush hour
- Sales spike during holidays

**In MC-ANN:** Captured using cos/sin date encoding and LSTM memory.

---

### Trend
**Simple:** Long-term increase or decrease in values.

**Analogy:** Like gradually gaining weight over years.

**Examples:**
- Climate change: temperatures increasing
- Reservoir: water levels declining due to drought

**In MC-ANN:** Handled through normalization and LSTM's ability to track long-term patterns.

---

### Stationarity
**Simple:** When statistical properties (mean, variance) don't change over time.

**Analogy:** Like a calm lake - water level stays roughly the same.

**Non-stationary Example:** Reservoir water level varies by season (not stationary).

**Why Important:** Many models assume stationarity.

**In MC-ANN:** Uses differencing to make data more stationary.

---

### Autocorrelation
**Simple:** How much current values depend on past values.

**Analogy:** If you're happy today, you're likely happy tomorrow (high autocorrelation).

**In Time Series:** Today's water level strongly correlates with yesterday's level.

**In MC-ANN:** LSTM exploits this by using past values to predict future.

---

### Extreme Events
**Simple:** Rare occurrences that deviate significantly from normal.

**Analogy:** Like a snowstorm in summer - very unusual.

**In MC-ANN:**
- **Floods:** Water level > 90th percentile
- **Droughts:** Water level < 10th percentile

**Challenge:** Rare (5-10% of data) but critical to predict.

**Solution:** Oversampling in training.

---

## 4. Model Architecture

### Architecture
**Simple:** The overall structure and design of the model.

**Analogy:** Like a building's blueprint - shows how different parts connect.

**In MC-ANN:**
```
Input → 3 LSTMs (Encoder) → Attention → 3 LSTMs (Decoder) → Output
```

---

### Attention Mechanism
**Simple:** A way for the model to focus on the most important parts of input.

**Analogy:** Like highlighting key sentences in a textbook - you pay more attention to them.

**Example:** When predicting flood, focus more on recent heavy rainfall periods.

**Types:**

**1. Self-Attention:**
**What:** Each part of input attends to all other parts.
**Analogy:** Reading a paragraph and understanding how each word relates to others.

**2. Cross-Attention:**
**What:** Output attends to input.
**Analogy:** When writing a summary (output), you reference the original text (input).

**In MC-ANN:** Uses self-attention to weight different GMM components.

---

### Multi-Head Attention
**Simple:** Multiple attention mechanisms running in parallel.

**Analogy:** Like having multiple people read the same text, each focusing on different aspects.

**In MC-ANN:** 4 heads - each learns different patterns.

**Benefit:** 
- Head 1 might focus on short-term patterns
- Head 2 might focus on long-term trends
- Head 3 might focus on extreme values
- Head 4 might focus on normal patterns

---

### Attention Weights
**Simple:** Numbers (0 to 1) showing how much focus to give each input.

**Analogy:** Like brightness levels on a dimmer switch - some inputs get more "light."

**In MC-ANN:**
- **w0:** Weight for low regime LSTM
- **w1:** Weight for medium regime LSTM  
- **w2:** Weight for high regime LSTM
- Sum: w0 + w1 + w2 = 1

**Example:**
- During flood: w2 = 0.7 (high regime dominates)
- During normal: w1 = 0.6 (medium regime dominates)
- During drought: w0 = 0.5 (low regime dominates)

---

### Positional Encoding / Embedding
**Simple:** Adding information about position in the sequence.

**Analogy:** Like adding page numbers to a book - you know where you are.

**Why Needed:** Neural networks don't inherently understand order.

**In MC-ANN:**
```python
pe[:, 0::2] = sin(position / 10000^(2i/d))
pe[:, 1::2] = cos(position / 10000^(2i/d))
```

**Effect:** Model knows timestep 1 is before timestep 2, etc.

---

### Embedding
**Simple:** Converting discrete values (words, categories) into continuous vectors.

**Analogy:** Like translating "dog" into coordinates [0.2, 0.8, 0.3] that capture meaning.

**Example:** 
- "cat" → [0.2, 0.7, 0.5]
- "dog" → [0.3, 0.8, 0.4]  (similar to cat)
- "car" → [0.9, 0.1, 0.2]  (very different)

**In MC-ANN:** Positional embeddings tell model about time positions.

---

### Dropout
**Simple:** Randomly "turning off" some neurons during training to prevent overfitting.

**Analogy:** Like practicing basketball with one hand tied - forces you to develop diverse skills.

**Example:** With 0.1 dropout, 10% of neurons randomly set to zero each training step.

**In MC-ANN:** Dropout = 0.1 in LSTM layers.

**Benefit:** 
- Prevents over-reliance on specific neurons
- Forces redundant learning
- Improves generalization

---

### Batch Normalization (BatchNorm)
**Simple:** Standardizing layer outputs to have mean=0, std=1.

**Analogy:** Like adjusting all test scores to the same scale before averaging.

**Why:** Stabilizes training, allows faster learning.

**In MC-ANN:** Used after attention layers.

**Formula:**
```
normalized = (x - mean) / sqrt(variance + ε)
output = γ × normalized + β
```

---

### Residual Connection / Skip Connection
**Simple:** Adding the input directly to the output of a layer.

**Analogy:** Like taking both the scenic route AND the highway - combine both paths.

**Example:**
```
output = Layer(input) + input
```

**In MC-ANN:** "Add & Norm" steps in attention module.

**Benefit:**
- Helps gradients flow during backpropagation
- Allows very deep networks
- Preserves information

---

### Mixture of Experts
**Simple:** Multiple specialized models, each expert in different scenarios.

**Analogy:** Like a hospital with specialists - cardiologist for heart, neurologist for brain.

**In MC-ANN:**
- Expert 0: Specializes in low water levels (drought)
- Expert 1: Specializes in medium water levels (normal)
- Expert 2: Specializes in high water levels (flood)

**How:** Attention mechanism chooses which expert(s) to trust.

---

## 5. Training & Optimization

### Loss Function
**Simple:** Measures how wrong the model's predictions are.

**Analogy:** Like counting mistakes on a test - more mistakes = higher score (bad).

**Goal:** Minimize loss = fewer mistakes.

**Common Loss Functions:**

**1. MSE (Mean Squared Error):**
```
MSE = average((prediction - truth)²)
```
**When:** Regression (predicting continuous values)
**In MC-ANN:** Primary training loss

**2. MAE (Mean Absolute Error):**
```
MAE = average(|prediction - truth|)
```
**When:** When you want to treat all errors equally

**3. Cross-Entropy:**
```
For classification (not used in MC-ANN)
```

---

### Optimizer
**Simple:** Algorithm that adjusts model parameters to reduce loss.

**Analogy:** Like a GPS finding the fastest route - tries different paths to find the best one.

**Common Optimizers:**

**1. Adam (Adaptive Moment Estimation):**
**Simple:** Smart optimizer that adapts learning rate for each parameter.
**In MC-ANN:** Used for both encoder and decoder.
**Why Popular:** Works well in most cases, requires less tuning.

**2. SGD (Stochastic Gradient Descent):**
**Simple:** Basic optimizer, updates based on gradients.

**3. RMSprop:**
**Simple:** Adapts learning rate based on recent gradients.

---

### Learning Rate
**Simple:** How big of a step the model takes when adjusting parameters.

**Analogy:** Like step size when walking - too large and you overshoot, too small and progress is slow.

**In MC-ANN:** 0.001 (starts) → decreases over time

**Visualization:**
```
Large LR (0.1):  🏃 Fast but may overshoot minimum
Medium LR (0.001): 🚶 Steady progress
Small LR (0.0001): 🐌 Slow but precise
```

**Trade-off:**
- **Too high:** Unstable training, loss bounces around
- **Too low:** Very slow training, may get stuck

---

### Learning Rate Schedule / Decay
**Simple:** Gradually reducing learning rate during training.

**Analogy:** Walking fast at first, then slowing down as you approach destination.

**In MC-ANN:** Type4 - exponential decay
```
LR = initial_LR × decay_factor^epoch
```

**Why:** Start with big steps (explore), end with small steps (fine-tune).

---

### Gradient
**Simple:** The direction and magnitude of change needed to reduce loss.

**Analogy:** Like slope of a hill - tells you which way is downhill and how steep.

**Mathematical:**
```
gradient = ∂Loss/∂parameter
```

**In Training:**
1. Calculate loss
2. Compute gradients (how each parameter affects loss)
3. Update parameters in opposite direction of gradient

---

### Backpropagation
**Simple:** Algorithm for computing gradients in neural networks.

**Analogy:** Like tracing back through your work to find where you made a mistake.

**Process:**
1. Forward pass: Input → predictions
2. Calculate loss
3. Backward pass: Propagate error back through layers
4. Compute gradient for each parameter

**In MC-ANN:** Happens automatically with PyTorch's `.backward()`.

---

### Epoch
**Simple:** One complete pass through the entire training dataset.

**Analogy:** Reading a textbook once from cover to cover.

**In MC-ANN:** Train for max 50 epochs.

**Example:**
- Training set: 30,000 samples
- Batch size: 48
- Batches per epoch: 30,000 / 48 = 625
- One epoch = 625 forward passes + backward passes

---

### Batch
**Simple:** A subset of training data processed together.

**Analogy:** Instead of grading exams one at a time, grade them in stacks of 48.

**In MC-ANN:** Batch size = 48

**Why Use Batches:**
- **Faster:** Process multiple samples in parallel (GPU efficiency)
- **Stable:** Gradients averaged over batch are less noisy
- **Memory:** Can't fit entire dataset in memory

**Types:**
- **Batch Gradient Descent:** Use entire dataset (slow)
- **Stochastic:** Use 1 sample (noisy)
- **Mini-batch:** Use 32-64 samples (best trade-off) ✓

---

### Iteration
**Simple:** One forward pass + one backward pass on one batch.

**In MC-ANN:**
- 1 epoch = 625 iterations (30,000 samples / 48 batch size)
- 50 epochs = 31,250 iterations total

---

### Early Stopping
**Simple:** Stop training when validation performance stops improving.

**Analogy:** Stop practicing when you're no longer getting better (to avoid overtraining).

**In MC-ANN:**
- Monitor validation RMSE after each epoch
- If no improvement for 4 consecutive epochs → stop
- Saves best model before stopping

**Benefit:** Prevents overfitting, saves time.

---

### Validation Set
**Simple:** Data held out during training to check if model is generalizing.

**Analogy:** Like practice exams before the real exam.

**In MC-ANN:**
- 60 random time points from training period
- Never used for training
- Used to monitor overfitting and tune hyperparameters

**Difference from Test Set:** 
- Validation: Used during training to guide decisions
- Test: Used once at the end to evaluate final performance

---

### Test Set
**Simple:** Data completely unseen during training, used for final evaluation.

**Analogy:** The actual exam (not practice).

**In MC-ANN:**
- 2018-07-01 to 2019-07-01 (entire year)
- Never touched during training or validation
- Used only to report final metrics

**Why Separate:** Ensures model truly generalizes to new data.

---

### Training Set
**Simple:** Data used to train the model.

**Analogy:** The textbook and practice problems.

**In MC-ANN:**
- 1983-07-01 to 2018-06-30 (35 years)
- 20,000-40,000 samples randomly selected
- With oversampling of extreme events

---

### Regularization
**Simple:** Techniques to prevent overfitting.

**Analogy:** Like rules that prevent students from just memorizing answers.

**Types:**

**1. Dropout:** Randomly disable neurons
**2. L1/L2 Regularization:** Penalize large weights
**3. Early Stopping:** Stop when validation performance degrades
**4. Data Augmentation:** Add noise or transformations

**In MC-ANN:** Uses dropout (0.1) and early stopping.

---

### Convergence
**Simple:** When training loss stops decreasing significantly.

**Analogy:** Like climbing a mountain - you've reached the top (or a plateau).

**Signs:**
- Loss curve flattens
- Validation performance stabilizes
- Small changes in parameters

**In MC-ANN:** Usually converges around epoch 25-40.

---

## 6. Evaluation Metrics

### Metric
**Simple:** A number that measures model performance.

**Analogy:** Like a grade on a test - higher (or lower, depending on metric) is better.

---

### RMSE (Root Mean Square Error)
**Simple:** Average prediction error, with larger errors penalized more.

**Formula:**
```
RMSE = sqrt(mean((predicted - actual)²))
```

**Example:**
```
Predictions: [100, 150, 200]
Actual:      [110, 140, 210]
Errors:      [-10, +10, -10]
Squared:     [100, 100, 100]
Mean:        100
RMSE:        √100 = 10
```

**In MC-ANN:** Primary metric, **lower is better**.

**Units:** Same as target variable (e.g., cubic meters)

**Why Square:** Penalizes large errors more (error of 10 → 100, error of 20 → 400).

---

### MAE (Mean Absolute Error)
**Simple:** Average of absolute errors.

**Formula:**
```
MAE = mean(|predicted - actual|)
```

**Example:**
```
Predictions: [100, 150, 200]
Actual:      [110, 140, 210]
Errors:      [-10, +10, -10]
Absolute:    [10, 10, 10]
MAE:         10
```

**Difference from RMSE:** Treats all errors equally (no squaring).

**When to Use:** When you want to treat small and large errors similarly.

---

### MAPE (Mean Absolute Percentage Error)
**Simple:** Average prediction error as a percentage.

**Formula:**
```
MAPE = mean(|predicted - actual| / |actual|) × 100%
```

**Example:**
```
Predictions: [100, 150, 200]
Actual:      [110, 140, 210]
Errors:      [10, 10, 10]
Percentages: [9.1%, 7.1%, 4.8%]
MAPE:        7.0%
```

**In MC-ANN:** Secondary metric, **lower is better**.

**Benefit:** Scale-independent - can compare across different datasets.

**Problem:** Undefined when actual = 0.

---

### MSE (Mean Squared Error)
**Simple:** Average of squared errors.

**Formula:**
```
MSE = mean((predicted - actual)²)
```

**Relationship:** RMSE = √MSE

**In MC-ANN:** Used as loss function during training.

---

### Accuracy
**Simple:** Percentage of correct predictions.

**Formula:**
```
Accuracy = (correct predictions) / (total predictions) × 100%
```

**Note:** Not used in MC-ANN (for classification, not regression).

---

### R² (R-squared / Coefficient of Determination)
**Simple:** How much of the variance in data is explained by the model.

**Range:** 0 to 1 (1 is perfect)

**Formula:**
```
R² = 1 - (sum of squared errors) / (total variance)
```

**Interpretation:**
- R² = 0.9 → Model explains 90% of variance
- R² = 0.5 → Model explains 50% of variance

---

## 7. Data Processing

### Normalization
**Simple:** Scaling data to a standard range.

**Analogy:** Like converting all temperatures to Celsius so you can compare them.

**Why:** Neural networks learn better when inputs are similar scales.

**Types:**

**1. Min-Max Normalization:**
```
normalized = (x - min) / (max - min)
Result: 0 to 1
```

**2. Z-Score Normalization (Standardization):**
```
normalized = (x - mean) / std
Result: mean=0, std=1
```

**In MC-ANN:** Uses log + standardization.

---

### Standardization
**Simple:** Transform data to have mean=0 and standard deviation=1.

**Formula:**
```
standardized = (x - mean) / std
```

**Example:**
```
Data: [10, 20, 30, 40, 50]
Mean: 30
Std:  ~14.14
Standardized: [-1.41, -0.71, 0, 0.71, 1.41]
```

**In MC-ANN:** Applied after log transform.

---

### Denormalization
**Simple:** Converting normalized data back to original scale.

**Formula:**
```
original = normalized × std + mean
```

**In MC-ANN:** Done after prediction to get actual water levels.

---

### Log Transform
**Simple:** Taking logarithm of data.

**Formula:**
```
log_x = log(x + ε)
where ε = small constant (e.g., 1e-10)
```

**Why:**
- Reduces impact of extreme values
- Makes multiplicative relationships additive
- Stabilizes variance

**Example:**
```
Original: [1, 10, 100, 1000]
Log:      [0, 2.3, 4.6, 6.9]
```

**In MC-ANN:** Applied before standardization.

---

### Differencing
**Simple:** Subtracting previous value from current value.

**Formula:**
```
diff[t] = x[t] - x[t-1]
```

**Example:**
```
Original: [100, 105, 103, 110]
Diff:     [-, 5, -2, 7]
```

**Why:** 
- Makes data stationary
- Removes trends
- Model predicts change instead of absolute value

**In MC-ANN:** Used in preprocessing (see `diff_order_1`).

---

### Missing Data / NaN
**Simple:** Values that are not available in the dataset.

**NaN:** "Not a Number" - placeholder for missing data.

**Analogy:** Like blank spaces in a form.

**Causes:**
- Sensor malfunction
- Transmission errors
- Data corruption

**Handling Strategies:**

**1. Deletion:** Skip samples with missing values (MC-ANN uses this)
**2. Imputation:** Fill with mean, median, or predicted value
**3. Interpolation:** Estimate based on surrounding values

**In MC-ANN:** Sequences with any NaN are excluded from training/testing.

---

### Feature Engineering
**Simple:** Creating new useful features from raw data.

**Analogy:** Like a chef preparing ingredients before cooking.

**In MC-ANN:**
- Raw: Water level value
- Engineered:
  - Normalized value
  - Extreme score (from GMM)
  - Cluster probabilities
  - Time features (cos/sin date)

**Goal:** Give model more informative inputs.

---

### Feature Extraction
**Simple:** Identifying and selecting relevant patterns from data.

**Analogy:** Like extracting key points from a long article.

**In MC-ANN:** 
- GMMs extract cluster patterns
- LSTMs extract temporal patterns
- Attention extracts important timesteps

---

### Data Augmentation
**Simple:** Creating artificial training examples by modifying existing ones.

**Analogy:** Like practicing basketball from different court positions.

**Examples:**
- Images: Rotate, flip, crop
- Time Series: Add noise, scale, shift

**In MC-ANN:** Not traditional augmentation, but oversampling serves similar purpose.

---

### Oversampling
**Simple:** Increasing representation of rare examples in training data.

**Analogy:** Like studying harder problems more often before an exam.

**In MC-ANN:**
- Extreme events naturally occur ~5% of time
- Oversampling increases to ~40% of training data
- Done by sampling multiple windows around extreme points

**Why:** Model learns rare but important patterns.

---

### Undersampling
**Simple:** Decreasing representation of common examples.

**Analogy:** Studying easy problems less often.

**Not used in MC-ANN**, but opposite of oversampling.

---

### Class Imbalance
**Simple:** When some categories appear much more often than others.

**Example:** 95% normal days, 5% extreme events.

**Problem:** Model ignores rare class (just predicts "normal" always).

**Solution:** Oversampling (or class weights, or SMOTE).

---

## 8. Mathematical Operations

### Matrix
**Simple:** A rectangular array of numbers.

**Analogy:** Like a spreadsheet with rows and columns.

**Example:**
```
[1, 2, 3]
[4, 5, 6]
```
2 rows × 3 columns

**In Neural Networks:** Weights are stored as matrices.

---

### Vector
**Simple:** A one-dimensional array of numbers.

**Analogy:** A single row or column of a spreadsheet.

**Example:**
```
[1, 2, 3, 4]
```

**In MC-ANN:** Each timestep's features form a vector.

---

### Tensor
**Simple:** A multi-dimensional array of numbers.

**Analogy:** Like a stack of spreadsheets (3D), or even higher dimensions.

**Examples:**
- 1D tensor: [1, 2, 3] (vector)
- 2D tensor: Matrix
- 3D tensor: [Batch, Sequence, Features]
- 4D tensor: [Batch, Channels, Height, Width] (images)

**In MC-ANN:**
```
Input tensor: [48, 360, 8]
  48 samples in batch
  360 timesteps per sample
  8 features per timestep
```

---

### Matrix Multiplication
**Simple:** Combining two matrices using specific rules.

**Example:**
```
[1, 2]  ×  [5, 6]  =  [19, 22]
[3, 4]     [7, 8]     [43, 50]
```

**In Neural Networks:** Core operation for transforming data.

**Formula:**
```
output[i][j] = Σ(A[i][k] × B[k][j])
```

---

### Dot Product
**Simple:** Multiply corresponding elements and sum.

**Example:**
```
[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 32
```

**In Neural Networks:** How neurons compute weighted sums of inputs.

---

### Element-wise Operation
**Simple:** Apply operation to each element independently.

**Example:**
```
[1, 2, 3] + [4, 5, 6] = [5, 7, 9]
[1, 2, 3] × [4, 5, 6] = [4, 10, 18]
```

**Symbol:** ⊙ (element-wise multiplication)

**In MC-ANN:**
```
out = w0 ⊙ out0 + w1 ⊙ out1 + w2 ⊙ out2
```

---

### Broadcasting
**Simple:** Automatically expanding dimensions to match for operations.

**Example:**
```
[1, 2, 3] + 10 = [11, 12, 13]
(10 is "broadcast" to [10, 10, 10])
```

**In MC-ANN:** GMM probabilities broadcast across timesteps.

---

### Concatenation
**Simple:** Joining arrays together.

**Example:**
```
concat([1, 2], [3, 4]) = [1, 2, 3, 4]
```

**In MC-ANN:**
```
Features: concat(value, extreme_score, GMM_probs) = 8 features
```

---

### Reshape / Flatten
**Simple:** Changing the shape of an array without changing data.

**Example:**
```
Original: [1, 2, 3, 4, 5, 6] (shape: 6)
Reshape:  [[1, 2], [3, 4], [5, 6]] (shape: 3×2)
```

**Flatten:** Convert multi-dimensional to 1D
```
[[1, 2], [3, 4]] → [1, 2, 3, 4]
```

**In MC-ANN:** GMM reshapes sequences for clustering.

---

### Squeeze / Unsqueeze
**Simple:** Remove or add dimensions of size 1.

**Example:**
```
[1, 2, 3] (shape: 3)
unsqueeze → [[1], [2], [3]] (shape: 3×1)
squeeze → [1, 2, 3] (shape: 3)
```

**In MC-ANN:** Adjusting tensor dimensions for operations.

---

### Mean / Average
**Simple:** Sum of values divided by count.

**Formula:**
```
mean = (x1 + x2 + ... + xn) / n
```

**Example:**
```
mean([1, 2, 3, 4, 5]) = 15/5 = 3
```

---

### Standard Deviation (Std)
**Simple:** Measure of how spread out numbers are.

**Formula:**
```
std = sqrt(mean((x - mean)²))
```

**Example:**
```
Data: [1, 2, 3, 4, 5]
Mean: 3
Deviations: [-2, -1, 0, 1, 2]
Squared: [4, 1, 0, 1, 4]
Variance: mean([4,1,0,1,4]) = 2
Std: sqrt(2) ≈ 1.41
```

**Interpretation:**
- Low std: Data clustered around mean
- High std: Data spread out

---

### Variance
**Simple:** Average of squared deviations from mean.

**Formula:**
```
variance = mean((x - mean)²)
```

**Relationship:** std = √variance

---

### Median
**Simple:** Middle value when data is sorted.

**Example:**
```
[1, 2, 3, 4, 5] → median = 3
[1, 2, 3, 4] → median = (2+3)/2 = 2.5
```

**Robust to Outliers:** Unlike mean, not affected by extreme values.

---

### Percentile
**Simple:** Value below which a percentage of data falls.

**Example:**
- 90th percentile: 90% of data is below this value
- 10th percentile: 10% of data is below this value

**In MC-ANN:**
- Flood: > 90th percentile
- Drought: < 10th percentile

---

### Softmax
**Simple:** Converts numbers to probabilities that sum to 1.

**Formula:**
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**Example:**
```
Input: [1.0, 2.0, 3.0]
Exp: [2.72, 7.39, 20.09]
Sum: 30.2
Softmax: [0.09, 0.24, 0.67]
```

**In MC-ANN:** Converts attention scores to weights.

---

### Exponential (exp)
**Simple:** e raised to a power (e ≈ 2.718).

**Formula:**
```
exp(x) = e^x
```

**Example:**
```
exp(0) = 1
exp(1) = 2.718
exp(2) = 7.389
```

**Used in:** Softmax, sigmoid activation.

---

### Logarithm (log)
**Simple:** Inverse of exponential.

**Formula:**
```
if y = e^x, then log(y) = x
```

**Example:**
```
log(1) = 0
log(2.718) = 1
log(7.389) = 2
```

**In MC-ANN:** Log transform of water levels.

---

## 9. Programming & Tools

### Python
**Simple:** A programming language popular for machine learning.

**Analogy:** Like English for humans, Python for computers.

**In MC-ANN:** All code written in Python 3.8.8.

---

### PyTorch
**Simple:** A Python library for building neural networks.

**Analogy:** Like LEGO for machine learning - provides building blocks.

**Features:**
- Tensor operations (like NumPy but with GPU support)
- Automatic differentiation (computes gradients)
- Pre-built layers (LSTM, Linear, etc.)

**In MC-ANN:** Core framework for model implementation.

---

### CUDA
**Simple:** NVIDIA's platform for GPU computing.

**Analogy:** Like a turbo engine for your car.

**Why:** GPUs process many calculations in parallel (much faster than CPU).

**In MC-ANN:** Optional but recommended (5-10x speedup).

---

### GPU (Graphics Processing Unit)
**Simple:** Hardware originally for graphics, now used for ML training.

**Analogy:** Like having 1000 slow workers vs 10 fast workers - parallel tasks done faster.

**In MC-ANN:**
- Training on GPU: ~5 min/epoch
- Training on CPU: ~30 min/epoch

---

### CPU (Central Processing Unit)
**Simple:** The main processor in your computer.

**Analogy:** The "brain" of the computer.

**For MC-ANN:** Can train on CPU, just slower.

---

### Tensor (in PyTorch)
**Simple:** PyTorch's version of multi-dimensional arrays (like NumPy arrays but with GPU support).

**Example:**
```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
x = x.to('cuda')  # Move to GPU
```

---

### NumPy
**Simple:** Python library for numerical computations with arrays.

**Analogy:** Like a scientific calculator for programming.

**Example:**
```python
import numpy as np
x = np.array([1, 2, 3])
mean = np.mean(x)
```

**In MC-ANN:** Used for data preprocessing.

---

### Pandas
**Simple:** Python library for data manipulation (like Excel in code).

**Analogy:** Spreadsheet software for programmers.

**Example:**
```python
import pandas as pd
df = pd.read_csv('data.tsv', sep='\t')
df['value'].mean()
```

**In MC-ANN:** Loading and processing TSV files.

---

### Scikit-learn (sklearn)
**Simple:** Python library for machine learning algorithms.

**Features:**
- Clustering (GMM)
- Preprocessing (normalization)
- Metrics (RMSE, MAE)

**In MC-ANN:** Used for Gaussian Mixture Models.

---

### Matplotlib
**Simple:** Python library for creating plots and visualizations.

**Analogy:** Like drawing tools for data.

**In MC-ANN:** Plotting predictions vs ground truth.

---

### Conda / Anaconda
**Simple:** Package and environment manager for Python.

**Analogy:** Like a toolbox organizer - keeps different project tools separate.

**Why:** Avoid conflicts between different package versions.

**In MC-ANN:**
```bash
conda create -n MCANN python=3.8.8
conda activate MCANN
```

---

### Virtual Environment
**Simple:** Isolated Python environment for a project.

**Analogy:** Like having separate toolboxes for different hobbies.

**Benefit:** Each project has its own package versions.

---

### Repository / Repo
**Simple:** A folder containing code and version history.

**In MC-ANN:** The GitHub repo contains all code, docs, and history.

---

### Git
**Simple:** Version control system (tracks changes to code over time).

**Analogy:** Like "Track Changes" in Microsoft Word, but for code.

**Commands:**
```bash
git add .          # Stage changes
git commit -m ""   # Save snapshot
git push          # Upload to GitHub
```

---

### GitHub
**Simple:** Website for hosting Git repositories.

**Analogy:** Like Google Drive for code.

**In MC-ANN:** Code hosted at github.com/davidanastasiu/mcann

---

### Jupyter Notebook
**Simple:** Interactive document mixing code, text, and visualizations.

**Analogy:** Like a lab notebook - document experiments with live code.

**In MC-ANN:** `example.ipynb` shows usage examples.

---

### Terminal / Command Line
**Simple:** Text-based interface to control your computer.

**Analogy:** Like chatting with your computer using typed commands.

**Example:**
```bash
cd /path/to/folder
python run.py
ls -l
```

---

### Script
**Simple:** A file containing code that performs a task.

**In MC-ANN:**
- `run.py`: Training script
- `predict.py`: Prediction script
- `test.py`: Testing script

---

### Module / Library / Package
**Simple:** Reusable code organized in files.

**Analogy:** Like ingredients you keep in your pantry.

**Example:**
```python
import torch           # Library
from models.GMM_Model5 import EncoderLSTM  # Module
```

---

### API (Application Programming Interface)
**Simple:** A way to interact with software programmatically.

**Analogy:** Like a restaurant menu - you don't need to know how food is cooked, just order from the menu.

**In MC-ANN:**
```python
model = DAN(opt, ds)
model.train()
```

---

### Checkpoint
**Simple:** Saved snapshot of model during training.

**Analogy:** Like save points in a video game.

**In MC-ANN:** Best model saved as `.zip` file.

**Contents:**
- Encoder weights
- Decoder weights
- GMM models
- Hyperparameters

---

### Pickle
**Simple:** Python's way of saving objects to files.

**Analogy:** Like freezing food to preserve it.

**In MC-ANN:** GMM models saved using pickle.

---

### TSV (Tab-Separated Values)
**Simple:** Text file where columns are separated by tabs.

**Example:**
```
datetime            value
2019-01-01 00:00   1234.5
2019-01-01 01:00   1235.2
```

**In MC-ANN:** Dataset format.

---

## 10. MC-ANN Specific Terms

### MC-ANN
**Simple:** Mixture Clustering-Based Attention Neural Network.

**Breakdown:**
- **Mixture:** Multiple models (experts) working together
- **Clustering:** Grouping similar patterns
- **Attention:** Focusing on important parts
- **Neural Network:** Deep learning model

---

### GMM (Gaussian Mixture Model)
**Simple:** A statistical model that assumes data comes from multiple Gaussian (bell-curve) distributions.

**Analogy:** Like assuming people's heights come from multiple groups (children, women, men) - each with its own average and spread.

**Visual:**
```
      *           *  *
    * | *       * |  | *
   *  |  *     *  |  |  *
  *   |   *   *   |  |   *
*─────┼─────*─────┼──┼─────*
   Low    Medium   High
```

**In MC-ANN:**
- 3 components (Gaussians)
- Represents low/medium/high water levels

**Output:** Probability of belonging to each component.

---

### EM Algorithm (Expectation-Maximization)
**Simple:** Algorithm to fit GMM to data.

**Steps:**
1. **E-step:** Assign data points to clusters (soft assignment)
2. **M-step:** Update cluster parameters
3. Repeat until convergence

**Analogy:** Like sorting socks:
1. Guess which socks match
2. Update your idea of what each pair looks like
3. Resort based on new understanding
4. Repeat

---

### Posterior Probability
**Simple:** Probability of belonging to a cluster given the data.

**Formula:**
```
P(cluster | data) = P(data | cluster) × P(cluster) / P(data)
```

**Example:**
- Water level = 1500
- P(high cluster | 1500) = 0.8
- P(medium cluster | 1500) = 0.15
- P(low cluster | 1500) = 0.05

**In MC-ANN:** GMM outputs posterior probabilities.

---

### Prior Probability
**Simple:** Probability before seeing any data (based on frequency).

**Example:**
- High cluster appears 20% of time → Prior = 0.2
- Medium cluster appears 60% → Prior = 0.6
- Low cluster appears 20% → Prior = 0.2

**In MC-ANN:** GMM weights are priors.

---

### Component (in GMM)
**Simple:** One Gaussian distribution in the mixture.

**In MC-ANN:** 3 components
- Component 0: Low water levels
- Component 1: Medium water levels
- Component 2: High water levels

**Each component has:**
- Mean (μ): Center
- Variance (σ²): Spread
- Weight (π): How often it occurs

---

### Regime
**Simple:** A distinct operating mode or pattern in the data.

**Analogy:** Like weather seasons - summer regime (hot), winter regime (cold).

**In MC-ANN:**
- Low regime: Drought conditions
- Medium regime: Normal conditions
- High regime: Flood conditions

---

### Extreme Score
**Simple:** Measure of how unusual a data point is.

**Formula:**
```
extreme_score = 1 - P(normal)
```

**Example:**
- Normal water level: extreme_score = 0.1 (10% unusual)
- Very high level: extreme_score = 0.9 (90% unusual)

**In MC-ANN:** Computed by GM3, used as input feature.

---

### Sequence-Level Clustering
**Simple:** Classifying entire sequences (not individual points).

**Analogy:** Like categorizing movies - not by individual scenes, but the whole movie.

**In MC-ANN:** GMM classifies last 72 timesteps together.

---

### Temporal Features
**Simple:** Features derived from time information.

**In MC-ANN:**
- Cos(date): Cyclical time encoding
- Sin(date): Cyclical time encoding

**Why:** Helps model understand "January is close to December."

---

### Cyclical Encoding
**Simple:** Representing circular time (hour, month) as coordinates.

**Problem with Simple Numbers:**
```
Hour: 23 → 0 → 1
Looks far apart (23 vs 0), but they're only 1 hour apart!
```

**Solution with Cos/Sin:**
```
Hour 23: (cos(2π×23/24), sin(2π×23/24)) ≈ (0.99, -0.26)
Hour 0:  (cos(0), sin(0)) = (1, 0)
Hour 1:  (cos(2π×1/24), sin(2π×1/24)) ≈ (0.99, 0.26)
```
Now 23 and 1 are close!

---

### DAN (Deep Attention Network)
**Simple:** The main training class in MC-ANN.

**Not an acronym in the paper**, just the class name for the overall system.

---

### Inference Module (MCANN_I)
**Simple:** Standalone class for making predictions with trained model.

**Purpose:** Load model and predict without needing training code.

---

### Roll
**Simple:** Stride between test predictions.

**In MC-ANN:** roll = 8
- Predict at hour 0, 8, 16, 24, 32...
- Not every hour (too slow)

---

### Stride
**Simple:** Step size when moving through data.

**Analogy:** Like reading every 5th page of a book instead of every page.

**In MC-ANN:** Determines how often to make test predictions.

---

## Summary Tables

### Quick Reference: Metrics

| Metric | Lower/Higher Better | Scale | Use Case |
|--------|---------------------|-------|----------|
| RMSE | Lower | Same as target | Primary metric |
| MAE | Lower | Same as target | Robust to outliers |
| MAPE | Lower | Percentage | Compare datasets |
| MSE | Lower | Squared units | Loss function |

### Quick Reference: Layers

| Layer Type | Input Shape | Output Shape | Purpose |
|------------|-------------|--------------|---------|
| LSTM | [B, 360, 2] | [B, 360, 512] | Sequence processing |
| Linear | [B, 512] | [B, 1] | Dimension reduction |
| Attention | [B, 72, 150] | [B, 72, 1] | Weighting |
| BatchNorm | [B, 72, X] | [B, 72, X] | Stabilization |

### Quick Reference: Data Splits

| Split | Time Range | Purpose | Size |
|-------|------------|---------|------|
| Train | 1983-2018 | Learning | 20K-40K samples |
| Val | Random from train | Monitoring | 60 samples |
| Test | 2018-2019 | Evaluation | ~10K samples |

---

## Analogies Summary

| Concept | Real-World Analogy |
|---------|-------------------|
| Neural Network | Team of workers, each doing simple tasks |
| Training | Practicing piano repeatedly |
| Inference | Performing after practice |
| LSTM | Notebook for writing/referencing notes |
| Attention | Highlighting important textbook sections |
| Encoder | Summarizing a book into cliff notes |
| Decoder | Reconstructing story from cliff notes |
| Dropout | Practicing with handicap to learn better |
| Learning Rate | Step size when walking |
| Batch | Grading exams in stacks |
| Epoch | Reading textbook cover-to-cover once |
| Oversampling | Studying hard problems more often |
| GMM | Assuming heights come from multiple groups |

---

## Learning Path

### Beginner → Understand These First:
1. Machine Learning
2. Neural Network
3. Training vs Inference
4. Time Series
5. Forecasting

### Intermediate → Then Learn:
1. LSTM
2. Encoder-Decoder
3. Attention
4. Loss Function
5. Optimizer

### Advanced → Finally Master:
1. GMM
2. Mixture of Experts
3. Backpropagation
4. Regularization
5. Hyperparameter Tuning

---

## Common Confusions Clarified

### Parameters vs Hyperparameters
- **Parameters:** Learned by model (weights, biases)
- **Hyperparameters:** Set by humans (learning rate, hidden size)

### Training vs Validation vs Test
- **Training:** Used to learn
- **Validation:** Used to tune and monitor during training
- **Test:** Used once at end to report final performance

### Overfitting vs Underfitting
- **Overfitting:** Too complex, memorizes training data
- **Underfitting:** Too simple, misses patterns

### Normalization vs Standardization
- **Normalization:** Scale to [0, 1]
- **Standardization:** Transform to mean=0, std=1

### RMSE vs MAE
- **RMSE:** Penalizes large errors more (squared)
- **MAE:** Treats all errors equally

---

## Questions You Might Have

**Q: Why 3 components in GMM?**
A: Trial and error showed 3 captures patterns well (low/medium/high). Could try 2, 4, 5...

**Q: Why 360 input timesteps?**
A: 15 days captures weekly and bi-weekly patterns. Shorter misses patterns; longer is slow.

**Q: Why 72 output timesteps?**
A: 3 days is useful for planning. Longer predictions become less accurate.

**Q: Why use LSTM instead of Transformer?**
A: LSTMs work well for moderate-length sequences with less data. Transformers need huge datasets.

**Q: Why oversampling instead of class weights?**
A: Provides model with more diverse examples of extreme events, not just higher loss penalty.

---

**End of Glossary**

**Remember:** Don't try to memorize everything! Use this as a reference when you encounter unfamiliar terms. Understanding comes with practice and experimentation.

**Next Steps:**
1. Read PROJECT_OVERVIEW.md while referring to this glossary
2. Try running the code
3. Come back here when you see confusing terms
4. Experiment and learn by doing!
