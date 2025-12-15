# Social Touch Research – Dataset Augmentation & LSTM Training

This repository contains the full pipeline for **augmenting touch-gesture datasets** and **training an LSTM-based classifier** on spatiotemporal touch grid data.

The intended workflow is:
1. Start from annotated gesture recordings
2. Generate an augmented dataset
3. Train and evaluate an LSTM model

---

## Repository Structure

```
.
├── Annotated-dataset/
│   └── Original labeled gesture recordings (JSON)
│
├── Original-recordings-dataset/
│   └── Raw / unprocessed recordings
│
├── Augment_dataset - file based.py
│   └── Dataset augmentation script
│
├── LSTM - file split - augmented class balance.py
│   └── LSTM training using original + augmented data
│
├── LSTM - file split - class balance.py
│   └── LSTM training using only original data
```

---

## Data Format

Each gesture recording is stored as a JSON file with the following structure:

```json
{
  "id": "unique_sample_id",
  "label": "gesture_name",
  "frames": [
    {
      "frame_index": 0,
      "label": "gesture_name",
      "data": [[...], [...], ...]
    }
  ]
}
```

- Each frame is a **19 × 27** touch grid
- Values are expected in the range **0–255**
- Labels are used at the sequence level

---

## Step 1 – Dataset Augmentation

### Script
```
Augment_dataset - file based.py
```

### What it does
- Loads annotated gesture recordings
- Applies configurable spatial and temporal augmentations
- Saves:
  - Processed original samples
  - Multiple augmented versions per sample

All output is written to:
```
augmented_dataset/
```

### How to run

```bash
python "Augment_dataset - file based.py"
```

---

### Key augmentation settings

Edit these inside `GestureDataAugmenter.__init__()`:

#### Number of augmentations
```python
self.num_augmentations_per_sample = 15
```

#### Spatial shifts (percentage of grid size)
```python
self.max_x_shift_percent = 0.5
self.max_y_shift_percent = 0.5
```

#### Rotation
```python
self.max_rotation_degrees = 0
```

#### Flipping
```python
self.flip_probability = 0.0
```

#### Speed variation
```python
self.enable_speed_variation = False
self.min_speed_factor = 0.8
self.max_speed_factor = 1.2
```

#### Scaling
```python
self.enable_scaling = False
```

#### Elastic deformation
```python
self.enable_elastic_deformation = False
```

#### Trimming based on contact area
```python
self.enable_trimming = False
```

---

### Augmentation output

Each augmented sample retains a reference to its original recording:

```json
"parent_id": "original_sample_id"
```

---

## Step 2 – LSTM Training (With Augmentation)

### Script
```
LSTM - file split - augmented class balance.py
```

This is the **primary training script** and can be run directly.

### How to run

```bash
python "LSTM - file split - augmented class balance.py"
```

---

### What the training script does

1. Loads original gesture samples
2. Splits data by **file ID** into:
   - Train (60%)
   - Validation (20%)
   - Test (20%)
3. Loads augmented samples **only for training IDs**
4. Creates sliding window sequences
5. Trains a CNN + LSTM model
6. Evaluates against a baseline (guess factor)
7. Saves the trained model and label encoder

---

### Key parameters you can change

#### Sliding window
```python
window_size = 5
stride = 3
```

#### Training
```python
iters = 5        # number of epochs
batch_size = 32
```

---

### Model architecture

- TimeDistributed CNN (32 → 64 filters, 5×5)
- MaxPooling
- LSTM (32 units)
- Dense softmax classification layer

This architecture captures **spatial patterns per frame** and **temporal dynamics across frames**.

---

### Baseline (Guess Factor)

The script computes a baseline accuracy by always predicting the most frequent class in the dataset.

Your model is expected to **beat this baseline**. A warning is printed if it does not.

---

### Outputs

After training, the following files are generated:

```
gesture_lstm_model_weighted.h5
label_encoder.pkl
```

Additionally, the script produces:
- Accuracy and loss plots
- Confusion matrix
- Classification report

---

## Step 3 – LSTM Training (Without Augmentation)

### Script
```
LSTM - file split - class balance.py
```

This script follows the same pipeline as the augmented version, but:
- Uses **only original data**
- Does **not** load augmented samples

Useful for:
- Ablation studies
- Comparing the effect of augmentation

---

## Installation

Install required dependencies:

```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow keras scipy
```

Optional (for elastic deformation):
```bash
pip install opencv-python
```

---

## Typical Workflow

```text
1. Place data in Annotated-dataset/
2. Run Augment_dataset - file based.py
3. Verify augmented_dataset/ is populated
4. Run LSTM - file split - augmented class balance.py
5. Inspect results and saved model
```

---

## Notes

- Ensure `window_size` is smaller than the number of frames per recording
- Augmented data is **never used** for validation or testing
- Git root must be opened in VS Code for commits to work

---

## License

This repository is intended for research and experimental use.

