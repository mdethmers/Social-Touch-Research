#!/usr/bin/python3
import json
import os
import sys  # Added for safe exiting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight 
from sklearn.metrics import confusion_matrix, classification_report
from keras import models, layers, losses
from collections import defaultdict, Counter

# --- HELPER: CALCULATE GUESS FACTOR ---
def calculate_guess_factor(y_data, name="Dataset"):
    if len(y_data) == 0:
        return 0.0
    
    counts = Counter(y_data)
    most_common_class, count = counts.most_common(1)[0]
    baseline_acc = count / len(y_data)
    
    print(f"[{name}] GUESS FACTOR (Baseline):")
    print(f"   Most frequent class: '{most_common_class}'")
    print(f"   Baseline Accuracy:   {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    return baseline_acc

# --- 1. SAFE DATASET CREATION ---
def create_dataset_safe(sections_list, window_size=5, stride=1):
    X_accumulator = []
    y_accumulator = []
    
    for section in sections_list:
        frames = section['frames']
        if len(frames) < window_size:
            continue
            
        section_data = np.array([f["data"] for f in frames])
        section_labels = [f["label"] for f in frames]
        
        for i in range(0, len(section_data) - window_size + 1, stride):
            window_data = section_data[i : i + window_size]
            label = section_labels[i + window_size - 1]
            X_accumulator.append(window_data)
            y_accumulator.append(label)
            
    return np.array(X_accumulator), np.array(y_accumulator)

def save_model(model, filepath: str) -> None:
    model.save(filepath)
    print(f"Model saved at: {filepath}")

# --- 2. DATA LOADING & SPLITTING ---
# CHECK THIS: Ensure this folder name matches your actual folder!
data_directory = 'augmented_dataset'  # Changed back to 'augmented_dataset'
sections_per_file = 1 

if not os.path.exists(data_directory):
    print(f"ERROR: Directory '{data_directory}' not found.")
    sys.exit(1)

all_json_files = [f for f in os.listdir(data_directory) if f.endswith('.json')]
print(f"Found {len(all_json_files)} total JSON files")

if len(all_json_files) == 0:
    print("ERROR: No JSON files found. Cannot train.")
    sys.exit(1)

original_files = [f for f in all_json_files if '_original.json' in f]
augmented_files = [f for f in all_json_files if '_aug_' in f]

# Group ORIGINAL files by class
sections_by_class = defaultdict(list)

print("\nProcessing Original Files...")
for file_name in original_files:
    file_path = os.path.join(data_directory, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
        class_label = data['label']
        frames = data['frames']
        
        total_frames = len(frames)
        frames_per_section = total_frames // sections_per_file
        
        for i in range(sections_per_file):
            start_idx = i * frames_per_section
            end_idx = total_frames if i == sections_per_file - 1 else (i + 1) * frames_per_section
            
            section_data = {
                'label': class_label,
                'frames': frames[start_idx:end_idx],
                'file_id': data['id'] 
            }
            sections_by_class[class_label].append(section_data)

# Split Strategy
train_sections = []
val_sections = []
test_sections = []
train_file_ids = [] 

for class_name, sections in sections_by_class.items():    
    class_train_val, class_test = train_test_split(sections, test_size=0.2, random_state=42)
    class_train, class_val = train_test_split(class_train_val, test_size=0.25, random_state=42)
    
    train_sections.extend(class_train)
    val_sections.extend(class_val)
    test_sections.extend(class_test)
    
    for section in class_train:
        train_file_ids.append(section['file_id'])

print(f"\nOriginal file split:")
print(f"Train sections: {len(train_sections)}")
print(f"Validation sections: {len(val_sections)}")
print(f"Test sections: {len(test_sections)}")

# --- 3. LOADING AUGMENTED DATA ---
print("\nLoading augmented data (only for Training IDs)...")
augmented_train_sections = []

for file_name in augmented_files:
    file_path = os.path.join(data_directory, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
        parent_id = data.get('parent_id', '')
        
        if parent_id in train_file_ids:
            class_label = data['label']
            frames = data['frames']
            
            total_frames = len(frames)
            frames_per_section = total_frames // sections_per_file
            
            for i in range(sections_per_file):
                start_idx = i * frames_per_section
                end_idx = total_frames if i == sections_per_file - 1 else (i + 1) * frames_per_section
                
                section_data = {
                    'label': class_label,
                    'frames': frames[start_idx:end_idx],
                    'file_id': data['id'],
                    'parent_id': parent_id,
                    'augmented': True
                }
                augmented_train_sections.append(section_data)

all_train_sections = train_sections + augmented_train_sections
print(f"Total train sections (original + augmented): {len(all_train_sections)}")

# --- 4. PREPARING SLIDING WINDOWS ---
window_size = 5
stride = 3

print("\nCreating sliding window samples using SAFE method...")
X_train, y_train_strings = create_dataset_safe(all_train_sections, window_size=window_size, stride=stride)
X_val, y_val_strings     = create_dataset_safe(val_sections, window_size=window_size, stride=stride)
X_test, y_test_strings   = create_dataset_safe(test_sections, window_size=window_size, stride=stride)

# SAFEGUARD: Check if data exists before proceeding
if len(X_train) == 0:
    print("\nCRITICAL ERROR: Training set is empty!")
    print("This usually means the dataset folder is empty or the window_size is larger than your video clips.")
    sys.exit(1)

# --- 5. ENCODING & NORMALIZATION ---
le = LabelEncoder()
y_train = le.fit_transform(y_train_strings)
y_val = le.transform(y_val_strings)
y_test = le.transform(y_test_strings)

print(f"\nClasses: {le.classes_}")

# Normalize (0-1)
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape
dim_row = 19
dim_col = 27
X_train = X_train.reshape(X_train.shape[0], window_size, dim_row, dim_col, 1)
X_val = X_val.reshape(X_val.shape[0], window_size, dim_row, dim_col, 1)
X_test = X_test.reshape(X_test.shape[0], window_size, dim_row, dim_col, 1)

print(f"Train Shape: {X_train.shape}")
print(f"Val Shape:   {X_val.shape}")

# --- 6. CALCULATE BASELINES AND WEIGHTS ---

# A. Calculate Guess Factor (The Accuracy to Beat)
train_baseline = calculate_guess_factor(y_train_strings, "Train Set")
test_baseline = calculate_guess_factor(y_test_strings, "Test Set")

# B. Calculate Class Weights
print("\nCalculating class weights to handle imbalance...")
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print("Class weights (Applied to Loss):")
for i, weight in class_weight_dict.items():
    print(f"  {le.inverse_transform([i])[0]}: {weight:.4f}")

# --- 7. MODEL DEFINITION ---
# IMPORTANT: I restored the LSTM layer. Without it, you will get a Rank/Shape mismatch error.
num_classes = len(le.classes_)
model = models.Sequential([
    layers.TimeDistributed(
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'), #32 filters of 5x5 size)
        input_shape=(window_size, dim_row, dim_col, 1)
    ),
    layers.TimeDistributed(layers.Conv2D(64, (5, 5), activation='relu', padding='same')), #64 filters of 5x5 size)
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(32),
    layers.Dense(num_classes, activation='softmax') # --- Dense classification layer (10 in the paper, num_classes for you) ---
])

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# --- 8. TRAINING ---
iters = 5
batch_size = 32

print("\nStarting training with Class Weights...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=iters,
    batch_size=batch_size,
    class_weight=class_weight_dict, 
    shuffle=True,
    verbose=1
)

# --- 9. EVALUATION ---
# Plotting with Baseline Reference Line
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# Red dashed line for Guess Factor
plt.axhline(y=test_baseline, color='r', linestyle='--', alpha=0.7, label=f'Baseline (Guess Factor: {test_baseline:.2f})')
plt.title('Accuracy vs Baseline')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predictions
print("\nMaking predictions on test set...")
y_pred_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_pred_strings = le.inverse_transform(y_pred)

test_accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Final Verdict
if test_accuracy > test_baseline:
    print(f"✅ SUCCESS: Model beat the guess factor by {(test_accuracy - test_baseline)*100:.2f}%")
else:
    print(f"⚠️ WARNING: Model failed to beat the guess factor of {test_baseline:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_strings, y_pred_strings))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_strings, y_pred_strings)
unique_labels = sorted(list(set(y_test_strings) | set(y_pred_strings)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.title(f'Confusion Matrix - Test Acc: {test_accuracy:.4f}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45)
plt.show()

# Save
save_model(model, 'gesture_lstm_model_weighted.h5')

import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("Label encoder saved.")