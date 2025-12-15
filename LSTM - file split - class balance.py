#!/usr/bin/python3
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from keras import models, layers, losses
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict
from collections import Counter

# --- HELPER: SAFE WINDOWING ---
def create_dataset_safe(sections_list, window_size=5, stride=1):
    """
    Applies sliding window to each file/section INDIVIDUALLY.
    This prevents the model from learning fake transitions between
    unrelated files (the 'Boundary Effect').
    """
    X_accumulator = []
    y_accumulator = []
    
    for section in sections_list:
        frames = section['frames']
        
        # Skip if section is too short
        if len(frames) < window_size:
            continue
            
        # Extract data and labels
        # Assuming frame["data"] is the image/sensor data
        section_data = np.array([f["data"] for f in frames])
        section_labels = [f["label"] for f in frames]
        
        # Create windows strictly within this section
        for i in range(0, len(section_data) - window_size + 1, stride):
            window_data = section_data[i : i + window_size]
            
            # Label is the label of the last frame in the window
            label = section_labels[i + window_size - 1]
            
            X_accumulator.append(window_data)
            y_accumulator.append(label)
            
    return np.array(X_accumulator), np.array(y_accumulator)

def guess_factor_check(labels, set_name="Dataset"):
    counts = Counter(labels)
    if not counts:
        return 0.0, "None"
    most_common = counts.most_common(1)[0]
    accuracy = most_common[1] / len(labels)
    print(f"[{set_name}] Baseline Accuracy (Guess Factor): {accuracy:.4f} (Majority Class: {most_common[0]})")
    return accuracy

def save_model(model, filepath: str) -> None:
    model.save(filepath)
    print(f"Model saved at: {filepath}")

# --- LOAD DATASET ---
data_directory = 'annotated_dataset'
sections_per_file = 1 

json_files = [f for f in os.listdir(data_directory) if f.endswith('.json')]
print(f"Found {len(json_files)} JSON files")

# Group file sections by class
sections_by_class = defaultdict(list)

print("Processing files (Preserving full timeline)...")
for file_name in json_files:
    file_path = os.path.join(data_directory, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
        class_label = data['label']
        frames = data['frames']

        # --- CRITICAL CHANGE ---
        # REMOVED the "None Reduction" logic here.
        # We keep all None frames to preserve the real speed of gestures.
        # We will handle the imbalance using Class Weights later.

        # Split frames into sections
        total_frames = len(frames)
        frames_per_section = total_frames // sections_per_file
        
        for i in range(sections_per_file):
            start_idx = i * frames_per_section
            end_idx = total_frames if i == sections_per_file - 1 else (i + 1) * frames_per_section
            
            section_data = {
                'label': class_label,
                'frames': frames[start_idx:end_idx],
                'file_name': file_name
            }
            sections_by_class[class_label].append(section_data)

print("Sections per class:")
for class_name, sections in sections_by_class.items():
    print(f"{class_name}: {len(sections)} sections")

# --- SPLITTING ---
train_sections = []
val_sections = []
test_sections = []

for class_name, sections in sections_by_class.items():    
    # First split: separate test sections (20%)
    class_train_val, class_test = train_test_split(
        sections, test_size=0.2, random_state=42 
    )
    
    # Second split: separate train/validation sections (25% of remaining)
    class_train, class_val = train_test_split(
        class_train_val, test_size=0.25, random_state=42 
    )
    
    train_sections.extend(class_train)
    val_sections.extend(class_val)
    test_sections.extend(class_test)

print(f"\nTrain sections: {len(train_sections)}")
print(f"Validation sections: {len(val_sections)}")
print(f"Test sections: {len(test_sections)}")

# --- WINDOWING (SAFE METHOD) ---
window_size = 5
stride = 3

print("\nCreating sliding window samples (Safe Mode)...")
X_train, y_train_strings = create_dataset_safe(train_sections, window_size=window_size, stride=stride)
X_val, y_val_strings     = create_dataset_safe(val_sections, window_size=window_size, stride=stride)
X_test, y_test_strings   = create_dataset_safe(test_sections, window_size=window_size, stride=stride)

# --- ENCODING ---
le = LabelEncoder()
le.fit(y_train_strings)
y_train = le.transform(y_train_strings)
y_val = le.transform(y_val_strings)
y_test = le.transform(y_test_strings)

print(f"\nX_train shape: {X_train.shape}")
print(f"Label classes: {le.classes_}")

# --- NORMALIZATION & RESHAPE ---
# Normalize
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape
dim_row = 19
dim_col = 27
X_train = X_train.reshape(X_train.shape[0], window_size, dim_row, dim_col, 1)
X_val = X_val.reshape(X_val.shape[0], window_size, dim_row, dim_col, 1)
X_test = X_test.reshape(X_test.shape[0], window_size, dim_row, dim_col, 1)

# --- CLASS WEIGHTS ---
print("\nComputing Class Weights...")
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print("Weights applied to loss function:")
for i, weight in class_weight_dict.items():
    print(f"  {le.inverse_transform([i])[0]}: {weight:.4f}")

# --- GUESS FACTOR CHECK ---
print("\n--- BASELINE CHECKS ---")
guess_factor_check(y_train_strings, "Training Set")
test_baseline = guess_factor_check(y_test_strings, "Test Set (Score to Beat)")
print("-----------------------")

# --- MODEL DEFINITION (Model 3) ---
num_classes = len(le.classes_)
model = models.Sequential([
    layers.TimeDistributed(
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        input_shape=(window_size, dim_row, dim_col, 1)
    ),
    layers.TimeDistributed(layers.Conv2D(64, (5, 5), activation='relu', padding='same')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(32),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

print("\nModel architecture:")
model.summary()

# --- TRAINING ---
iters = 5
batch_size = 32

print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=iters,
    batch_size=batch_size,
    class_weight=class_weight_dict, 
    shuffle=True,
    verbose=1
)

# --- EVALUATION ---
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_baseline, color='r', linestyle='--', label='Baseline (Guess Factor)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.tight_layout()
plt.show()

# Predictions
print("\nMaking predictions on test set...")
y_pred_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_pred_strings = le.inverse_transform(y_pred)

# Accuracy
test_accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
if test_accuracy > test_baseline:
    print(f"SUCCESS: Model beat the baseline by {(test_accuracy - test_baseline)*100:.2f}%")
else:
    print(f"WARNING: Model did not beat the baseline of {test_baseline:.4f}")

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test_strings, y_pred_strings)
unique_labels = sorted(list(set(y_test_strings) | set(y_pred_strings)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy:.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test_strings, y_pred_strings))

save_model(model, 'gesture_lstm_model.h5')

# Final Stats
print(f"\nDataset Statistics:")
print(f"Total samples: {len(X_train) + len(X_val) + len(X_test)}")
print(f"Samples per class (Train): {dict(zip(le.classes_, np.bincount(y_train)))}")

print("\nTraining completed!")