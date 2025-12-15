#!/usr/bin/env python3
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time
import glob
from scipy.ndimage import shift, rotate
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class GestureDataAugmenter:
    def __init__(self, input_dir="annotated_dataset", output_dir="augmented_dataset"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.gesture_counts = {}
        self.rows, self.columns = 19, 27  # Default dimensions based on original data

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure augmentation parameters
        self.num_augmentations_per_sample = 15
        self.max_x_shift_percent = 0.5  # Maximum shift as percentage of dimensions
        self.max_y_shift_percent = 0.5  # Maximum shift as percentage of dimensions
        self.max_rotation_degrees = 0  # Maximum rotation in degrees
        self.flip_probability = 0.0  # Probability of flipping the sample
        self.filter_threshold = 0  # Threshold for filtering low values
        
        # Speed variation parameters
        self.enable_speed_variation = False
        self.min_speed_factor = 0.8
        self.max_speed_factor = 1.2
        
        # Scaling parameters
        self.enable_scaling = False
        self.min_scale_x = 0.8
        self.max_scale_x = 1.2
        self.min_scale_y = 0.8
        self.max_scale_y = 1.2
        
        # Elastic deformation parameters
        self.enable_elastic_deformation = False
        self.deformation_alpha = 25
        self.deformation_sigma = 20

        # Trimming parameters
        self.enable_trimming = False
        self.trim_area_threshold = 0.01
        self.trim_frames_to_remove = 0
        self.trim_min_frames = 3
    
    def load_sample(self, filepath):
        """Load a single sample from a JSON file"""
        try:
            with open(filepath, 'r') as file:
                sample = json.load(file)
                return sample
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_frame_data(self, sample):
        """Extract frame data arrays from the nested structure"""
        frames = []
        for frame_obj in sample["frames"]:
            frames.append(np.array(frame_obj["data"]))
        return frames
    
    def reconstruct_frame_structure(self, frame_arrays, original_sample):
        """Reconstruct the frame structure with data arrays"""
        reconstructed_frames = []
        
        # Handle case where augmentation changed the number of frames
        original_frame_count = len(original_sample["frames"])
        new_frame_count = len(frame_arrays)
        
        for i, frame_array in enumerate(frame_arrays):
            # Map to original frame if possible
            if i < original_frame_count:
                original_frame = original_sample["frames"][i]
                frame_label = original_frame.get("label", "None")
            else:
                # For interpolated frames, use label from nearest original frame
                frame_label = original_sample["frames"][-1].get("label", "None")
            
            reconstructed_frames.append({
                "frame_index": i,
                "label": frame_label,
                "data": frame_array
            })
        
        return reconstructed_frames
            
    def _apply_speed_variation(self, frames, speed_factor):
        """Vary the speed of the gesture sequence by interpolating or dropping frames"""
        if speed_factor == 1.0:
            return frames
        
        num_frames = len(frames)
        
        if speed_factor < 1.0:  # Slow down, add frames
            new_num_frames = int(num_frames / speed_factor)
            indices = np.linspace(0, num_frames - 1, new_num_frames)
            new_frames = []
            
            for i in indices:
                if i.is_integer():
                    new_frames.append(frames[int(i)])
                else:
                    i_floor = int(np.floor(i))
                    i_ceil = int(np.ceil(i))
                    weight_ceil = i - i_floor
                    weight_floor = 1 - weight_ceil
                    
                    interpolated_frame = weight_floor * frames[i_floor] + weight_ceil * frames[i_ceil]
                    new_frames.append(interpolated_frame)
            
            return np.array(new_frames)
        
        else:  # Speed up, drop frames
            new_num_frames = max(3, int(num_frames * speed_factor))
            indices = np.linspace(0, num_frames - 1, new_num_frames)
            indices = indices.astype(int)
            
            return frames[indices]

    def _apply_scaling(self, frame, scale_x, scale_y):
        """Apply independent scaling in x and y directions"""
        from scipy.ndimage import zoom
        
        height, width = frame.shape
        new_height = max(3, int(height * scale_y))
        new_width = max(3, int(width * scale_x))
        
        try:
            zoomed = zoom(frame, (new_height/height, new_width/width), order=1)
            result = np.zeros((height, width), dtype=frame.dtype)
            
            y_offset = max(0, (height - new_height) // 2)
            x_offset = max(0, (width - new_width) // 2)
            
            if new_height <= height and new_width <= width:
                result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = zoomed
            else:
                crop_y_offset = max(0, (new_height - height) // 2)
                crop_x_offset = max(0, (new_width - width) // 2)
                
                result = zoomed[
                    crop_y_offset:crop_y_offset+height, 
                    crop_x_offset:crop_x_offset+width
                ]
                
                if result.shape != (height, width):
                    result = zoom(result, (height/result.shape[0], width/result.shape[1]), order=1)
            
            return result
        except Exception as e:
            print(f"Error in scaling: {e}")
            return frame

    def _apply_elastic_deformation(self, frame, alpha, sigma):
        """Apply elastic deformation to a frame"""
        try:
            import cv2
            
            height, width = frame.shape
            dx = np.random.rand(height, width) * 2 - 1
            dy = np.random.rand(height, width) * 2 - 1
            
            dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
            
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            
            map_x = np.float32(x + dx)
            map_y = np.float32(y + dy)
            
            deformed = cv2.remap(frame.astype(np.float32), map_x, map_y, 
                            interpolation=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_REFLECT_101)
            
            return deformed
        except Exception as e:
            print(f"Error in deformation: {e}, falling back to original frame")
            return frame

    def _apply_filter_threshold(self, frames):
        """Apply filter threshold to frames"""
        if self.filter_threshold > 0:
            filtered_frames = []
            for frame in frames:
                frame_array = np.array(frame)
                filtered_frame = np.where(frame_array < self.filter_threshold, 0, frame_array)
                filtered_frame = np.clip(filtered_frame, 0, 255)
                filtered_frame = np.round(filtered_frame, 1)
                filtered_frames.append(filtered_frame)
            return filtered_frames
        return frames

    def _trim_frames_by_contact_area(self, frames):
        """Trim frames from beginning and end based on contact area threshold."""
        if not self.enable_trimming or not frames:
            return frames
        
        contact_areas = []
        for frame in frames:
            frame_array = np.array(frame)
            frame_max = np.max(frame_array)
            if frame_max > 0:
                contact_threshold = 0.5 * frame_max
                contact_area = np.sum(frame_array > contact_threshold) / (self.rows * self.columns)
                contact_areas.append(contact_area)
            else:
                contact_areas.append(0.0)
        
        start_index = 0
        for i, area in enumerate(contact_areas):
            if area > self.trim_area_threshold:
                start_index = max(0, i - self.trim_frames_to_remove)
                break
        
        end_index = len(frames)
        for i in range(len(contact_areas) - 1, -1, -1):
            if contact_areas[i] > self.trim_area_threshold:
                end_index = min(len(frames), i + 1 + self.trim_frames_to_remove)
                break
        
        if end_index - start_index < self.trim_min_frames:
            max_area_index = np.argmax(contact_areas)
            half_min = self.trim_min_frames // 2
            start_index = max(0, max_area_index - half_min)
            end_index = min(len(frames), start_index + self.trim_min_frames)
            if end_index > len(frames):
                end_index = len(frames)
                start_index = max(0, end_index - self.trim_min_frames)
        
        return frames[start_index:end_index]
    
    def augment_frames(self, frames, augmentation_id):
        """Apply augmentation transformations to a set of frames with cylindrical wrapping"""
        frames_array = np.array(frames)
        augmented_frames = frames_array.copy()
        
        shift_y = random.uniform(-self.max_y_shift_percent, self.max_y_shift_percent) * self.rows
        shift_x = random.uniform(-self.max_x_shift_percent, self.max_x_shift_percent) * self.columns
        rotation_angle = random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
        do_flip = random.random() < self.flip_probability
        
        speed_factor = 1.0
        scale_x = 1.0
        scale_y = 1.0
        apply_deformation = False
        
        if self.enable_speed_variation and random.random() > 0.6:
            speed_factor = random.uniform(self.min_speed_factor, self.max_speed_factor)
        
        if self.enable_scaling and random.random() > 0.6:
            scale_x = random.uniform(self.min_scale_x, self.max_scale_x)
            scale_y = random.uniform(self.min_scale_y, self.max_scale_y)
        
        if self.enable_elastic_deformation and random.random() > 0.7:
            apply_deformation = True
        
        if speed_factor != 1.0:
            augmented_frames = self._apply_speed_variation(augmented_frames, speed_factor)
        
        for i in range(len(augmented_frames)):
            frame = augmented_frames[i]
            
            if scale_x != 1.0 or scale_y != 1.0:
                frame = self._apply_scaling(frame, scale_x, scale_y)
            
            if apply_deformation:
                frame = self._apply_elastic_deformation(frame, self.deformation_alpha, self.deformation_sigma)
            
            frame = np.roll(frame, int(round(shift_x)), axis=1)
            
            from scipy.ndimage import shift as ndshift
            frame = ndshift(frame, (int(round(shift_y)), 0), 
                        mode='constant', cval=0)
            
            frame = rotate(frame, rotation_angle, 
                        reshape=False, mode='constant', cval=0)
            
            if do_flip:
                frame = np.fliplr(frame)
                
            frame = np.clip(frame, 0, 255)
            frame = np.round(frame, 1)
            
            augmented_frames[i] = frame
        
        aug_params = {
            "shift_y": float(shift_y),
            "shift_x": float(shift_x),
            "rotation_angle": float(rotation_angle),
            "flipped": do_flip,
            "speed_factor": float(speed_factor),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "applied_deformation": apply_deformation,
            "augmentation_id": augmentation_id
        }
        
        return augmented_frames, aug_params

    def augment_dataset(self):
        """Augment all JSON files in the input directory"""
        print("Starting dataset augmentation...")
        start_time = time.time()
        
        # Print augmentation parameters
        print(f"Applying {self.num_augmentations_per_sample} augmentations per sample")
        print(f"- Filter threshold: {self.filter_threshold}")
        print(f"- Shift: X={self.max_x_shift_percent*100}%, Y={self.max_y_shift_percent*100}%")
        print(f"- Rotation: ±{self.max_rotation_degrees}°")
        print(f"- Flip probability: {self.flip_probability*100}%")
        print(f"- Speed variation: {'Enabled' if self.enable_speed_variation else 'Disabled'}, " 
            f"factor range: {self.min_speed_factor} to {self.max_speed_factor}")
        print(f"- Scaling: {'Enabled' if self.enable_scaling else 'Disabled'}, "
            f"X: {self.min_scale_x} to {self.max_scale_x}, Y: {self.min_scale_y} to {self.max_scale_y}")
        print(f"- Elastic deformation: {'Enabled' if self.enable_elastic_deformation else 'Disabled'}")
        print(f"- Trimming: {'Enabled' if self.enable_trimming else 'Disabled'}")
        
        # Get all JSON files
        json_files = glob.glob(os.path.join(self.input_dir, "*.json"))
        print(f"\nFound {len(json_files)} files to process\n")
        
        total_samples = 0
        
        for filepath in json_files:
            filename = os.path.basename(filepath)
            print(f"Processing {filename}...")
            
            # Load sample
            sample = self.load_sample(filepath)
            if sample is None:
                continue
            
            # Extract frame data
            frames = self.extract_frame_data(sample)
            
            # Apply filter threshold
            filtered_frames = self._apply_filter_threshold(frames)
            
            # Apply trimming (skip for "none" gestures)
            if sample["label"].lower() != "none" and self.enable_trimming:
                trimmed_frames = self._trim_frames_by_contact_area(filtered_frames)
            else:
                trimmed_frames = filtered_frames
            
            # Save original (processed) sample with filter applied
            original_output = sample.copy()
            # Convert trimmed frames back to lists for JSON serialization
            trimmed_frames_list = [frame.tolist() if isinstance(frame, np.ndarray) else frame 
                                   for frame in trimmed_frames]
            original_output["frames"] = self.reconstruct_frame_structure(trimmed_frames_list, sample)
            original_output["original_frame_count"] = len(frames)
            original_output["trimmed_frame_count"] = len(trimmed_frames)
            
            output_filename = f"{sample['id']}_original.json"
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(original_output, f, cls=NumpyEncoder, indent=2)
            
            total_samples += 1
            
            # Update gesture counts
            gesture = sample["label"]
            self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1
            
            # Generate augmented versions
            for i in range(self.num_augmentations_per_sample):
                aug_id = f"{i+1}"
                augmented_frames, aug_params = self.augment_frames(trimmed_frames, aug_id)
                
                # Create augmented sample
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                augmented_sample = sample.copy()
                augmented_sample["id"] = f"{sample['id']}_aug_{aug_id}"
                augmented_sample["timestamp"] = timestamp
                augmented_sample["parent_id"] = sample["id"]
                augmented_sample["augmentation_params"] = aug_params
                augmented_sample["frames"] = self.reconstruct_frame_structure(augmented_frames, sample)
                
                # Save augmented sample
                aug_filename = f"{augmented_sample['id']}_{timestamp}.json"
                aug_path = os.path.join(self.output_dir, aug_filename)
                
                with open(aug_path, 'w') as f:
                    json.dump(augmented_sample, f, cls=NumpyEncoder, indent=2)
                
                total_samples += 1
                self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1
        
        elapsed_time = time.time() - start_time
        print(f"\nAugmentation completed in {elapsed_time:.2f} seconds")
        print(f"Total samples created: {total_samples}")
        
        print("\nGesture counts:")
        for gesture, count in sorted(self.gesture_counts.items()):
            print(f"  {gesture}: {count}")

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that supports NumPy arrays and other numeric types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    # Create augmenter instance
    augmenter = GestureDataAugmenter(
        input_dir="annotated_dataset",
        output_dir="augmented_dataset"
    )
    
    # Perform augmentation
    augmenter.augment_dataset()