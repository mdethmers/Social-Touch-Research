import time
import cv2
import numpy as np
import serial
from collections import deque
import forcedimension_core.dhd as fdsdk
import json
import os
from datetime import datetime

#------------Modified to record animations with gesture and personality labels------------#
#added recording+preview of servo PWM values

# Configuration
WINDOW_WIDTH = 1600  # Increased width as requested
WINDOW_HEIGHT = 600
SCROLL_SPEED = 20.0  # ADJUST THIS: pixels per second scrolling speed (higher = faster scrolling)
DATA_POINTS_PER_SECOND = 10  # Keep 10 data points per second
DISPLAY_FPS = 20  # Display update rate
MAX_POINTS = int(WINDOW_WIDTH / SCROLL_SPEED * DATA_POINTS_PER_SECOND)  # Calculate based on scroll speed
SMOOTH_WINDOW = 1
ENABLE_SMOOTHING = False # Set to True to enable smoothing when outputting Serial
RAW_DATA_SIZE = 1000
WINDOW_TIME = 90

# Serial Configuration
SERIAL_PORT = 'COM18'  # Change to your ESP's port (e.g., '/dev/ttyUSB0' on Linux)
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 0.01  # Shorter timeout for faster response
DEBUG_MODE = False  # Set to False to disable debug prints
ENABLE_SERIAL_OUTPUT = True  # Set to False to disable robot serial communication

# Recording Configuration
RECORDINGS_DIR = "robot_animations"
GESTURES_FILE = "gestures.json"
PERSONALITIES_FILE = "personalities.json"

# Servo Configuration - Customize each servo independently
SERVO_CONFIG = {
    'height': {
        'index': 2,        # ESP32 servo index (0, 1, 2, etc.)
        'pwm_min': 500,    # Minimum PWM value
        'pwm_max': 2500,   # Maximum PWM value
        'pwm_center': 1500, # Center/neutral PWM value
        'inverse': False
    },
    'pitch': {
        'index': 1,        # ESP32 servo index
        'pwm_min': 500,    # Different range for this servo
        'pwm_max': 2500,   
        'pwm_center': 1350,
        'inverse': False
    },
    'roll': {
        'index': 0,        # ESP32 servo index
        'pwm_min': 800,    # Another different range
        'pwm_max': 2500,
        'pwm_center': 1520,  # Different center point
        'inverse': True
    }
}

# Customizable limits for haptic device (input ranges)
Z_MIN = -10.0  # cm
Z_MAX = 10.0   # cm
PITCH_MIN = -90.0  # degrees
PITCH_MAX = 90.0   # degrees
ROLL_MIN = -90.0   # degrees
ROLL_MAX = 90.0    # degrees

# Data storage (raw data)
z_data_raw = deque(maxlen=RAW_DATA_SIZE)
pitch_data_raw = deque(maxlen=RAW_DATA_SIZE)
roll_data_raw = deque(maxlen=RAW_DATA_SIZE)

# Data storage (smoothed data for plotting)
z_data = deque(maxlen=MAX_POINTS)
pitch_data = deque(maxlen=MAX_POINTS)
roll_data = deque(maxlen=MAX_POINTS)

# Add after the existing global variables
show_height_graph = True
show_pitch_graph = True
show_roll_graph = True

# Recording variables
recording_data = []
recording_start_time = None

# GUI state variables
current_gesture_index = 0
current_personality_index = 0
gui_mode = "NORMAL"  # NORMAL, EDIT_GESTURES, EDIT_PERSONALITIES
input_text = ""
edit_index = -1
serial_enabled = ENABLE_SERIAL_OUTPUT  # Runtime toggle for serial output

# Playback and scaling variables
recording_state = "IDLE"  # IDLE, RECORDING, PREVIEWING
playback_data = []
playback_start_time = None
playback_index = 0
movement_scale = 1.0  # Scale factor for movements (0.1 to 3.0)
saved_animations = []  # List of saved animation files
selected_animation_index = 0
loaded_animation_data = None

# Colors (BGR format for OpenCV)
BG_COLOR = (50, 50, 50)
GRID_COLOR = (80, 80, 80)
Z_COLOR = (0, 0, 255)      # Red for height
PITCH_COLOR = (255, 0, 0)  # Blue for pitch
ROLL_COLOR = (0, 255, 0)   # Green for roll
TEXT_COLOR = (255, 255, 255)
RECORDING_COLOR = (0, 0, 255)  # Red for recording indicator
BUTTON_COLOR = (100, 100, 100)
BUTTON_ACTIVE_COLOR = (150, 150, 150)

class LabelManager:
    def __init__(self):
        self.gestures = self.load_list(GESTURES_FILE, ["grab", "hug", "punch", "squeez", "stroke", "tickle", "tap", "personality"])
        self.personalities = self.load_list(PERSONALITIES_FILE, ["excited", "calm", "defensive"])
        
    def load_list(self, filename, default_list):
        """Load list from file or create with defaults"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
            else:
                self.save_list(filename, default_list)
                return default_list
        except:
            return default_list
    
    def save_list(self, filename, data_list):
        """Save list to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(data_list, f, indent=2)
        except Exception as e:
            print(f"Error saving {filename}: {e}")
    
    def save_all(self):
        """Save both lists"""
        self.save_list(GESTURES_FILE, self.gestures)
        self.save_list(PERSONALITIES_FILE, self.personalities)
    
    def add_gesture(self, gesture):
        """Add new gesture"""
        if gesture and gesture not in self.gestures:
            self.gestures.append(gesture)
            self.save_all()
    
    def add_personality(self, personality):
        """Add new personality"""
        if personality and personality not in self.personalities:
            self.personalities.append(personality)
            self.save_all()
    
    def remove_gesture(self, index):
        """Remove gesture by index"""
        if 0 <= index < len(self.gestures):
            self.gestures.pop(index)
            self.save_all()
    
    def remove_personality(self, index):
        """Remove personality by index"""
        if 0 <= index < len(self.personalities):
            self.personalities.pop(index)
            self.save_all()

def load_saved_animations():
    """Load list of saved animation files"""
    global saved_animations
    saved_animations = []

    if DEBUG_MODE:
        print("DEBUG: load_saved_animations() called - this will clear playback_data") 

    
    if os.path.exists(RECORDINGS_DIR):
        for filename in os.listdir(RECORDINGS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(RECORDINGS_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        animation_info = {
                            'filename': filename,
                            'filepath': filepath,
                            'gesture': data['metadata'].get('gesture', 'unknown'),
                            'personality': data['metadata'].get('personality', 'unknown'),
                            'duration': data['metadata'].get('duration', 0),
                            'data': data['data']
                        }
                        saved_animations.append(animation_info)
                except:
                    continue
    
    saved_animations.sort(key=lambda x: x['filename'])

def interpolate_playback_data(data, timestamp):
    """Interpolate between recorded data points based on timestamp"""
    if not data or len(data) < 2:
        if DEBUG_MODE:
            print(f"DEBUG: interpolate_playback_data - insufficient data. len(data)={len(data) if data else 0}")
        return None
    
    # Find the two data points to interpolate between
    for i in range(len(data) - 1):
        if data[i]['timestamp'] <= timestamp <= data[i + 1]['timestamp']:
            t1, t2 = data[i]['timestamp'], data[i + 1]['timestamp']
            if t2 - t1 == 0:  # Avoid division by zero
                return data[i]
            
            # Linear interpolation factor
            factor = (timestamp - t1) / (t2 - t1)
            
            result = {
                'height_pwm': int(data[i]['height_pwm'] + factor * (data[i + 1]['height_pwm'] - data[i]['height_pwm'])),
                'pitch_pwm': int(data[i]['pitch_pwm'] + factor * (data[i + 1]['pitch_pwm'] - data[i]['pitch_pwm'])),
                'roll_pwm': int(data[i]['roll_pwm'] + factor * (data[i + 1]['roll_pwm'] - data[i]['roll_pwm'])),
                'height_cm': data[i]['height_cm'] + factor * (data[i + 1]['height_cm'] - data[i]['height_cm']),
                'pitch_deg': data[i]['pitch_deg'] + factor * (data[i + 1]['pitch_deg'] - data[i]['pitch_deg']),
                'roll_deg': data[i]['roll_deg'] + factor * (data[i + 1]['roll_deg'] - data[i]['roll_deg'])
            }
            if DEBUG_MODE:
                print(f"DEBUG: interpolated at t={timestamp:.3f}: {result}")
            return result
    
    # If timestamp is beyond the data, return the last point
    if DEBUG_MODE:
        print(f"DEBUG: timestamp {timestamp:.3f} beyond data range, returning last point")
    return data[-1] if data else None

def update_playback(ser):
    """Update playback during PREVIEWING state"""
    global playback_start_time, playback_data, recording_state
    
    if DEBUG_MODE:
        print(f"DEBUG: update_playback called. recording_state={recording_state}, len(playback_data)={len(playback_data) if playback_data else 'None'}")
        print(f"DEBUG: update_playback - playback_data memory address: {id(playback_data) if playback_data else 'None'}")
        print(f"DEBUG: update_playback - playback_data type: {type(playback_data)}")
        print(f"DEBUG: update_playback - playback_data content: {playback_data[:2] if playback_data else 'None'}")
    
    
    if recording_state != "PREVIEWING" or not playback_data:
        print(f"DEBUG: Returning None because recording_state={recording_state} or playback_data is empty")
        return None
    
    if playback_start_time is None:
        playback_start_time = time.time()
        print(f"🎬 Starting playback with {len(playback_data)} data points")
    
    current_time = time.time() - playback_start_time
    animation_duration = playback_data[-1]['timestamp'] if playback_data else 0
    
    # Loop the animation
    if animation_duration > 0:
        loop_time = current_time % animation_duration
        interpolated = interpolate_playback_data(playback_data, loop_time)
        
        if interpolated:
            # Apply scaling to the preview values
            scaled_height, scaled_pitch, scaled_roll = apply_movement_scaling(
                interpolated['height_cm'], 
                interpolated['pitch_deg'], 
                interpolated['roll_deg']
            )
            
            # Recalculate PWM with scaled values
            scaled_height_pwm = map_to_pwm(scaled_height, Z_MIN, Z_MAX, SERVO_CONFIG['height'])
            scaled_pitch_pwm = map_to_pwm(scaled_pitch, PITCH_MIN, PITCH_MAX, SERVO_CONFIG['pitch'])
            scaled_roll_pwm = map_to_pwm(scaled_roll, ROLL_MIN, ROLL_MAX, SERVO_CONFIG['roll'])
            
            # Debug print to see if we're getting here
            print(f"🎭 Sending: H:{scaled_height_pwm} P:{scaled_pitch_pwm} R:{scaled_roll_pwm}")
            # Force send commands during preview, bypassing serial_enabled flag
            result = send_servo_commands(ser, scaled_height_pwm, scaled_pitch_pwm, scaled_roll_pwm, force_send=True)
            print(f"🔧 Send result: {result}")
            
            # Return scaled values for display
            interpolated['height_cm'] = scaled_height
            interpolated['pitch_deg'] = scaled_pitch
            interpolated['roll_deg'] = scaled_roll
            interpolated['height_pwm'] = scaled_height_pwm
            interpolated['pitch_pwm'] = scaled_pitch_pwm
            interpolated['roll_pwm'] = scaled_roll_pwm
        
        return interpolated
    
    return None

def apply_movement_scaling(height_cm, pitch_deg, roll_deg):
    """Apply scaling to movement values"""
    global movement_scale
    
    # Scale around center points
    height_center = (Z_MIN + Z_MAX) / 2
    pitch_center = (PITCH_MIN + PITCH_MAX) / 2  
    roll_center = (ROLL_MIN + ROLL_MAX) / 2
    
    scaled_height = height_center + (height_cm - height_center) * movement_scale
    scaled_pitch = pitch_center + (pitch_deg - pitch_center) * movement_scale
    scaled_roll = roll_center + (roll_deg - roll_center) * movement_scale
    
    # Clamp to limits
    scaled_height = max(Z_MIN, min(Z_MAX, scaled_height))
    scaled_pitch = max(PITCH_MIN, min(PITCH_MAX, scaled_pitch))
    scaled_roll = max(ROLL_MIN, min(ROLL_MAX, scaled_roll))
    
    return scaled_height, scaled_pitch, scaled_roll

def map_value(value, from_min, from_max, to_min, to_max):
    """Map a value from one range to another"""
    # Clamp input value to input range
    value = max(from_min, min(from_max, value))
    return int(to_min + (value - from_min) * (to_max - to_min) / (from_max - from_min))

def map_to_pwm(value, input_min, input_max, servo_config):
    """
    Map haptic device values to PWM, with center corresponding to input center
    When the haptic device is at center (0), PWM output will be pwm_center
    """
    # Calculate the center of the input range
    input_center = (input_min + input_max) / 2.0
    
    # Clamp input value to input range
    value = max(input_min, min(input_max, value))
    
    # Calculate the deviation from center
    deviation_from_center = value - input_center
    
    # Calculate the input range from center to max/min
    input_half_range = (input_max - input_min) / 2.0
    
    # Calculate PWM range from center to max/min
    pwm_half_range = (servo_config['pwm_max'] - servo_config['pwm_min']) / 2.0
    
    # Map the deviation proportionally
    if input_half_range > 0:
        pwm_deviation = deviation_from_center * (pwm_half_range / input_half_range)
    else:
        pwm_deviation = 0
    
    # Calculate final PWM value
    mapped_pwm = servo_config['pwm_center'] + pwm_deviation
    
    # Apply inverse at PWM level if configured
    if servo_config.get('inverse', False):
        # For inverse, flip around the center
        mapped_pwm = servo_config['pwm_center'] - pwm_deviation
    
    # Ensure PWM stays within bounds
    mapped_pwm = max(servo_config['pwm_min'], min(servo_config['pwm_max'], mapped_pwm))
    
    return int(mapped_pwm)

def moving_average(data_deque, window_size):
    """Calculate moving average of the last window_size elements"""
    if len(data_deque) < window_size:
        return sum(data_deque) / len(data_deque) if data_deque else 0.0
    else:
        recent_data = list(data_deque)[-window_size:]
        return sum(recent_data) / window_size

def add_smoothed_data(raw_value, raw_deque, smooth_deque):
    """Add raw data and calculate smoothed value"""
    raw_deque.append(raw_value)
    smoothed_value = moving_average(raw_deque, SMOOTH_WINDOW)
    smooth_deque.append(smoothed_value)
    return smoothed_value

def send_servo_commands(ser, height_pwm, pitch_pwm, roll_pwm, force_send=False):
    """Send PWM commands to ESP32 for 3 servos - separate messages"""
    global serial_enabled
    
    print(f"🚀 send_servo_commands called: H:{height_pwm} P:{pitch_pwm} R:{roll_pwm}, force_send:{force_send}, serial_enabled:{serial_enabled}")
    
    if (not serial_enabled and not force_send) or ser is None:
        print(f"⚠️ Skipping serial send: serial_enabled={serial_enabled}, force_send={force_send}, ser={ser}")
        return True  # Skip serial communication if disabled
    
    try:
        # Send each servo command using configured indices
        height_idx = SERVO_CONFIG['height']['index']
        pitch_idx = SERVO_CONFIG['pitch']['index'] 
        roll_idx = SERVO_CONFIG['roll']['index']
        
        ser.write(f"{height_idx},{height_pwm}\n".encode('utf-8'))
        time.sleep(0.001)  # 1ms delay between commands
        ser.write(f"{pitch_idx},{pitch_pwm}\n".encode('utf-8'))
        time.sleep(0.001)
        ser.write(f"{roll_idx},{roll_pwm}\n".encode('utf-8'))
        time.sleep(0.001)
        
        # Read any responses from ESP32 (non-blocking)
        while ser.in_waiting > 0:
            try:
                response = ser.readline().decode('utf-8').strip()
                if response and DEBUG_MODE:
                    print(f"ESP32: {response}")
            except:
                break
        
        return True
    except Exception as e:
        print(f"Serial communication error: {e}")
        return False

def start_recording(label_manager, current_gesture_index, current_personality_index):
    """Start recording animation data"""
    global recording_data, recording_state, recording_start_time
    
    if recording_state == "IDLE":
        recording_data = []
        recording_state = "RECORDING"
        recording_start_time = time.time()
        
        gesture = label_manager.gestures[current_gesture_index] if current_gesture_index < len(label_manager.gestures) else "unknown"
        personality = label_manager.personalities[current_personality_index] if current_personality_index < len(label_manager.personalities) else "unknown"
        
        print(f"🔴 RECORDING STARTED - Gesture: {gesture}, Personality: {personality}")

def stop_recording(label_manager, current_gesture_index, current_personality_index):
    """Stop recording and enter preview mode"""
    global recording_data, recording_state, playback_data, playback_start_time
    
    if recording_state == "RECORDING":
        recording_state = "PREVIEWING"
        playback_data = recording_data.copy()
        playback_start_time = None
        
        gesture = label_manager.gestures[current_gesture_index] if current_gesture_index < len(label_manager.gestures) else "unknown"
        personality = label_manager.personalities[current_personality_index] if current_personality_index < len(label_manager.personalities) else "unknown"
        
        print(f"⏯️ PREVIEWING - Press ENTER to save, R to re-record, ESC to cancel")

def save_recording(label_manager, current_gesture_index, current_personality_index):
    """Save the current recording and return to IDLE"""
    global recording_data, recording_state, playback_data, saved_animations
    
    if DEBUG_MODE:
        print("DEBUG: save_recording() called - this will clear playback_data")

    if recording_state == "PREVIEWING" and recording_data:
        # Create recordings directory if it doesn't exist
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        
        gesture = label_manager.gestures[current_gesture_index] if current_gesture_index < len(label_manager.gestures) else "unknown"
        personality = label_manager.personalities[current_personality_index] if current_personality_index < len(label_manager.personalities) else "unknown"
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{gesture}_{personality}_{timestamp}.json"
        filepath = os.path.join(RECORDINGS_DIR, filename)
        
        # Prepare data for saving
        animation_data = {
            "metadata": {
                "gesture": gesture,
                "personality": personality,
                "duration": recording_data[-1]['timestamp'] if recording_data else 0,
                "data_points": len(recording_data),
                "timestamp": timestamp,
                "servo_config": SERVO_CONFIG
            },
            "data": recording_data
        }
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(animation_data, f, indent=2)
            print(f"💾 Recording saved: {filepath}")
            print(f"   Duration: {animation_data['metadata']['duration']:.1f}s")
            print(f"   Data points: {len(recording_data)}")
            
            # Reload saved animations list - THIS WAS MISSING THE GLOBAL UPDATE
            load_saved_animations()
            
        except Exception as e:
            print(f"❌ Error saving recording: {e}")
    
    # Return to idle state
    recording_state = "IDLE"
    recording_data = []
    playback_data = []

def cancel_recording():
    """Cancel recording/preview and return to IDLE"""
    global recording_data, recording_state, playback_data
    
    if DEBUG_MODE:
        print("DEBUG: cancel_recording() called - this will clear playback_data") 

    recording_state = "IDLE"
    recording_data = []
    playback_data = []
    print("❌ Recording cancelled")

def record_data_point(height_cm, pitch_deg, roll_deg, height_pwm, pitch_pwm, roll_pwm):
    """Record a single data point if recording is active"""
    global recording_data, recording_start_time, recording_state
    
    if recording_state == "RECORDING" and recording_start_time:
        data_point = {
            "timestamp": time.time() - recording_start_time,
            "height_cm": height_cm,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "height_pwm": height_pwm,
            "pitch_pwm": pitch_pwm,
            "roll_pwm": roll_pwm
        }
        recording_data.append(data_point)

def draw_button(img, x, y, width, height, text, active=False):
    """Draw a button on the image"""
    color = BUTTON_ACTIVE_COLOR if active else BUTTON_COLOR
    cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), TEXT_COLOR, 1)
    
    # Center text in button
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    return (x, y, width, height)

def handle_mouse_click(event, x, y, flags, param):
    """Handle mouse clicks for GUI interaction"""
    global current_gesture_index, current_personality_index, gui_mode, input_text, edit_index, serial_enabled, movement_scale, selected_animation_index, loaded_animation_data, recording_state, playback_data, show_height_graph, show_pitch_graph, show_roll_graph
    label_manager = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Record/Save/Re-record button (150, 150, 100, 30)
        if 150 <= x <= 250 and 150 <= y <= 180:
            if recording_state == "IDLE":
                start_recording(label_manager, current_gesture_index, current_personality_index)
            elif recording_state == "RECORDING":
                stop_recording(label_manager, current_gesture_index, current_personality_index)
            elif recording_state == "PREVIEWING":
                save_recording(label_manager, current_gesture_index, current_personality_index)

        # Re-record button (only visible during preview) (270, 150, 100, 30)
        elif 270 <= x <= 370 and 150 <= y <= 180:
            if recording_state == "PREVIEWING":
                recording_state = "RECORDING"
                recording_data = []
                recording_start_time = time.time()
                print("🔄 Re-recording started")
            else:
                serial_enabled = not serial_enabled
                status = "ENABLED" if serial_enabled else "DISABLED"
                print(f"🔌 Serial output {status}")
        
        # Scale buttons (390, 150, 30, 30) and (430, 150, 30, 30)
        elif 390 <= x <= 420 and 150 <= y <= 180:  # Scale down
            movement_scale = max(0.1, movement_scale - 0.1)
            print(f"📏 Scale: {movement_scale:.1f}")
        elif 430 <= x <= 460 and 150 <= y <= 180:  # Scale up
            movement_scale = min(3.0, movement_scale + 0.1)
            print(f"📏 Scale: {movement_scale:.1f}")
        
        # Saved animation navigation (150, 270, 30, 30) and (400, 270, 30, 30)
        elif 150 <= x <= 180 and 270 <= y <= 300:  # Prev animation
            if saved_animations:
                selected_animation_index = (selected_animation_index - 1) % len(saved_animations)
        elif 400 <= x <= 430 and 270 <= y <= 300:  # Next animation
            if saved_animations:
                selected_animation_index = (selected_animation_index + 1) % len(saved_animations)
        
        elif 450 <= x <= 550 and 270 <= y <= 300:
            if saved_animations and recording_state == "IDLE":
                loaded_animation_data = saved_animations[selected_animation_index]['data']
                recording_state = "PREVIEWING"
                playback_data = loaded_animation_data.copy()
                playback_start_time = None

                if DEBUG_MODE:
                    print(f"DEBUG: Setting playback_data to {len(playback_data)} data points")
                    print(f"DEBUG: playback_data memory address: {id(playback_data)}")
                    print(f"DEBUG: playback_data is: {playback_data[:2] if playback_data else 'None'}")  # Show first 2 items
                    print(f"DEBUG: Loaded {len(playback_data)} data points")

                if playback_data:
                    if DEBUG_MODE:
                        print(f"DEBUG: First data point: {playback_data[0]}")
                        print(f"DEBUG: Last data point: {playback_data[-1]}")
                    # Check if all required fields exist
                    required_fields = ['timestamp', 'height_pwm', 'pitch_pwm', 'roll_pwm', 'height_cm', 'pitch_deg', 'roll_deg']
                    missing_fields = [field for field in required_fields if field not in playback_data[0]]
                    if missing_fields:
                        if DEBUG_MODE:
                            print(f"DEBUG: Missing fields in loaded data: {missing_fields}")
                    else:
                        if DEBUG_MODE:
                            print("DEBUG: All required fields present in loaded data")
                
                print(f"Loaded: {saved_animations[selected_animation_index]['filename']}")
        
        # Existing gesture/personality navigation buttons (adjust y positions)
        elif 150 <= x <= 180 and 180 <= y <= 210:  # Gesture prev
            current_gesture_index = (current_gesture_index - 1) % len(label_manager.gestures)
        elif 400 <= x <= 430 and 180 <= y <= 210:  # Gesture next
            current_gesture_index = (current_gesture_index + 1) % len(label_manager.gestures)
        
        elif 150 <= x <= 180 and 210 <= y <= 240:  # Personality prev
            current_personality_index = (current_personality_index - 1) % len(label_manager.personalities)
        elif 400 <= x <= 430 and 210 <= y <= 240:  # Personality next
            current_personality_index = (current_personality_index + 1) % len(label_manager.personalities)
        
        # Edit buttons (adjust y positions)
        elif 450 <= x <= 550 and 180 <= y <= 210:  # Edit gestures
            gui_mode = "EDIT_GESTURES"
            input_text = ""
        elif 450 <= x <= 550 and 210 <= y <= 240:  # Edit personalities
            gui_mode = "EDIT_PERSONALITIES"
            input_text = ""
        elif 150 <= x <= 200 and 310 <= y <= 335:  # Height toggle
            show_height_graph = not show_height_graph
            print(f"Height graph: {'ON' if show_height_graph else 'OFF'}")
        elif 210 <= x <= 260 and 310 <= y <= 335:  # Pitch toggle
            show_pitch_graph = not show_pitch_graph
            print(f"Pitch graph: {'ON' if show_pitch_graph else 'OFF'}")
        elif 270 <= x <= 320 and 310 <= y <= 335:  # Roll toggle
            show_roll_graph = not show_roll_graph
            print(f"Roll graph: {'ON' if show_roll_graph else 'OFF'}")

def clear_graph_data():
    """Clear all graph data points"""
    global z_data, pitch_data, roll_data, z_data_raw, pitch_data_raw, roll_data_raw
    
    z_data.clear()
    pitch_data.clear()
    roll_data.clear()
    z_data_raw.clear()
    pitch_data_raw.clear()
    roll_data_raw.clear()
    
    print("📊 Graph data cleared")

def draw_gui_controls(img, label_manager):
    """Draw GUI controls for recording and label management"""
    global gui_mode, input_text, serial_enabled, movement_scale, selected_animation_index
    
    # Recording button - changes based on state
    if recording_state == "IDLE":
        record_text = "START REC"
    elif recording_state == "RECORDING":
        record_text = "STOP REC"
    elif recording_state == "PREVIEWING":
        record_text = "SAVE"
    
    draw_button(img, 150, 150, 100, 30, record_text, recording_state != "IDLE")
    
    # Second button - changes based on state
    if recording_state == "PREVIEWING":
        draw_button(img, 270, 150, 100, 30, "RE-RECORD")
    else:
        serial_text = "SERIAL ON" if serial_enabled else "SERIAL OFF"
        draw_button(img, 270, 150, 100, 30, serial_text, serial_enabled)
    
    # Scale controls
    draw_button(img, 390, 150, 30, 30, "-")
    cv2.putText(img, f"{movement_scale:.1f}x", (470, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    draw_button(img, 430, 150, 30, 30, "+")
    cv2.putText(img, "Scale:", (470, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    
    cv2.putText(img, "Graph Visibility:", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # Height toggle
    height_color = Z_COLOR if show_height_graph else (80, 80, 80)
    draw_button(img, 150, 310, 50, 25, "Height", show_height_graph)
    cv2.rectangle(img, (145, 305), (205, 340), height_color, 2)
    
    # Pitch toggle  
    pitch_color = PITCH_COLOR if show_pitch_graph else (80, 80, 80)
    draw_button(img, 210, 310, 50, 25, "Pitch", show_pitch_graph)
    cv2.rectangle(img, (205, 305), (265, 340), pitch_color, 2)
    
    # Roll toggle
    roll_color = ROLL_COLOR if show_roll_graph else (80, 80, 80)
    draw_button(img, 270, 310, 50, 25, "Roll", show_roll_graph)
    cv2.rectangle(img, (265, 305), (325, 340), roll_color, 2)

    # Recording state indicator
    state_color = {
        "IDLE": (100, 100, 100),
        "RECORDING": RECORDING_COLOR,
        "PREVIEWING": (0, 255, 255)  # Yellow
    }
    cv2.circle(img, (130, 165), 8, state_color[recording_state], -1)
    cv2.putText(img, recording_state[:3], (105, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    
    # Serial status indicator (only show when not in preview mode)
    if recording_state != "PREVIEWING":
        serial_color = (0, 255, 0) if serial_enabled else (0, 0, 255)
        cv2.circle(img, (380, 165), 6, serial_color, -1)
    
    # Existing gesture/personality controls (same as before)
    gesture_text = label_manager.gestures[current_gesture_index] if current_gesture_index < len(label_manager.gestures) else "None"
    cv2.putText(img, "Gesture:", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    draw_button(img, 150, 180, 30, 30, "<")
    cv2.putText(img, f"{gesture_text}", (190, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    draw_button(img, 400, 180, 30, 30, ">")
    draw_button(img, 450, 180, 100, 30, "Edit Gestures")
    
    personality_text = label_manager.personalities[current_personality_index] if current_personality_index < len(label_manager.personalities) else "None"
    cv2.putText(img, "Personality:", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    draw_button(img, 150, 210, 30, 30, "<")
    cv2.putText(img, f"{personality_text}", (190, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    draw_button(img, 400, 210, 30, 30, ">")
    draw_button(img, 450, 210, 100, 30, "Edit Personalities")
    
    # Saved animations controls
    cv2.putText(img, "Saved Animations:", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    if saved_animations:
        animation_name = saved_animations[selected_animation_index]['filename'][:30] if len(saved_animations[selected_animation_index]['filename']) > 20 else saved_animations[selected_animation_index]['filename']
        draw_button(img, 150, 270, 30, 30, "<")
        cv2.putText(img, f"{animation_name}", (190, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        draw_button(img, 400, 270, 30, 30, ">")
        draw_button(img, 450, 270, 100, 30, "LOAD", recording_state == "IDLE")
        cv2.putText(img, f"({selected_animation_index + 1}/{len(saved_animations)})", (560, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    else:
        cv2.putText(img, "No saved animations", (190, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    
    # Show edit mode (existing code - adjust y position if needed)
    if gui_mode != "NORMAL":
        cv2.rectangle(img, (10, 310), (WINDOW_WIDTH - 10, 410), (40, 40, 40), -1)
        cv2.rectangle(img, (10, 310), (WINDOW_WIDTH - 10, 410), TEXT_COLOR, 1)
        
        if gui_mode == "EDIT_GESTURES":
            cv2.putText(img, "EDIT GESTURES", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            cv2.putText(img, "Current gestures:", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
            for i, gesture in enumerate(label_manager.gestures):
                cv2.putText(img, f"{i}: {gesture}", (20, 370 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        elif gui_mode == "EDIT_PERSONALITIES":
            cv2.putText(img, "EDIT PERSONALITIES", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            cv2.putText(img, "Current personalities:", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
            for i, personality in enumerate(label_manager.personalities):
                cv2.putText(img, f"{i}: {personality}", (20, 370 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        
        cv2.putText(img, f"Input: {input_text}_", (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        cv2.putText(img, "Type new item and press ENTER to add", (400, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        cv2.putText(img, "Type number and press DELETE to remove", (400, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        cv2.putText(img, "Press ESC to exit edit mode", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

def handle_keyboard_input(key, label_manager):
    """Handle keyboard input for editing labels"""
    global gui_mode, input_text, current_gesture_index, current_personality_index
    
    if gui_mode == "NORMAL":
        return
    
    if key == 27:  # ESC
        gui_mode = "NORMAL"
        input_text = ""
    elif key == 13:  # ENTER
        if input_text.strip():
            if gui_mode == "EDIT_GESTURES":
                label_manager.add_gesture(input_text.strip())
            elif gui_mode == "EDIT_PERSONALITIES":
                label_manager.add_personality(input_text.strip())
            input_text = ""
    elif key == 127 or key == 8:  # DELETE or BACKSPACE
        if input_text.isdigit():
            index = int(input_text)
            if gui_mode == "EDIT_GESTURES":
                if index < len(label_manager.gestures):
                    label_manager.remove_gesture(index)
                    if current_gesture_index >= len(label_manager.gestures):
                        current_gesture_index = max(0, len(label_manager.gestures) - 1)
            elif gui_mode == "EDIT_PERSONALITIES":
                if index < len(label_manager.personalities):
                    label_manager.remove_personality(index)
                    if current_personality_index >= len(label_manager.personalities):
                        current_personality_index = max(0, len(label_manager.personalities) - 1)
        input_text = ""
    elif 32 <= key <= 126:  # Printable characters
        input_text += chr(key)

def draw_plot(img):
    """Draw the real-time plot with adjustable scroll speed"""
    img.fill(50)
    
    # Draw grid lines
    for i in range(0, WINDOW_HEIGHT, 50):
        cv2.line(img, (0, i), (WINDOW_WIDTH, i), GRID_COLOR, 1)
    for i in range(0, WINDOW_WIDTH, 100):
        cv2.line(img, (i, 0), (i, WINDOW_HEIGHT), GRID_COLOR, 1)
    
    # Draw center line
    center_y = WINDOW_HEIGHT // 2
    cv2.line(img, (0, center_y), (WINDOW_WIDTH, center_y), (100, 100, 100), 2)
    
    # Calculate time window based on scroll speed
    time_window = WINDOW_WIDTH / SCROLL_SPEED
    
    # Draw time markers
    if len(z_data) > 0:
        marker_interval = max(5, int(time_window / 10))
        for sec in range(0, int(time_window) + 1, marker_interval):
            x_pos = int(WINDOW_WIDTH - (sec * SCROLL_SPEED))
            if 0 <= x_pos <= WINDOW_WIDTH:
                cv2.line(img, (x_pos, 0), (x_pos, WINDOW_HEIGHT), (120, 120, 120), 1)
                cv2.putText(img, f"-{sec}s", (x_pos + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    
    # Draw data if we have points
    if len(z_data) > 1:
        pixels_per_point = SCROLL_SPEED / DATA_POINTS_PER_SECOND
        
        # Draw Height (Z) - only if enabled
        if show_height_graph:
            for i in range(1, len(z_data)):
                x1 = int(WINDOW_WIDTH - (len(z_data) - i + 1) * pixels_per_point)
                x2 = int(WINDOW_WIDTH - (len(z_data) - i) * pixels_per_point)
                y1 = map_value(z_data[i-1], Z_MIN, Z_MAX, WINDOW_HEIGHT - 50, 50)
                y2 = map_value(z_data[i], Z_MIN, Z_MAX, WINDOW_HEIGHT - 50, 50)
                if x1 >= 0 and x2 >= 0:
                    cv2.line(img, (x1, y1), (x2, y2), Z_COLOR, 2)
        
        # Draw Pitch - only if enabled
        if show_pitch_graph:
            for i in range(1, len(pitch_data)):
                x1 = int(WINDOW_WIDTH - (len(pitch_data) - i + 1) * pixels_per_point)
                x2 = int(WINDOW_WIDTH - (len(pitch_data) - i) * pixels_per_point)
                y1 = map_value(pitch_data[i-1], PITCH_MIN, PITCH_MAX, WINDOW_HEIGHT - 50, 50)
                y2 = map_value(pitch_data[i], PITCH_MIN, PITCH_MAX, WINDOW_HEIGHT - 50, 50)
                if x1 >= 0 and x2 >= 0:
                    cv2.line(img, (x1, y1), (x2, y2), PITCH_COLOR, 2)
        
        # Draw Roll - only if enabled
        if show_roll_graph:
            for i in range(1, len(roll_data)):
                x1 = int(WINDOW_WIDTH - (len(roll_data) - i + 1) * pixels_per_point)
                x2 = int(WINDOW_WIDTH - (len(roll_data) - i) * pixels_per_point)
                y1 = map_value(roll_data[i-1], ROLL_MIN, ROLL_MAX, WINDOW_HEIGHT - 50, 50)
                y2 = map_value(roll_data[i], ROLL_MIN, ROLL_MAX, WINDOW_HEIGHT - 50, 50)
                if x1 >= 0 and x2 >= 0:
                    cv2.line(img, (x1, y1), (x2, y2), ROLL_COLOR, 2)
    
    # Draw labels and current values with PWM values (dim text if graph is hidden)
    if len(z_data) > 0:
        height_pwm = map_to_pwm(z_data[-1], Z_MIN, Z_MAX, SERVO_CONFIG['height'])
        pitch_pwm = map_to_pwm(pitch_data[-1], PITCH_MIN, PITCH_MAX, SERVO_CONFIG['pitch'])
        roll_pwm = map_to_pwm(roll_data[-1], ROLL_MIN, ROLL_MAX, SERVO_CONFIG['roll'])
        
        height_color = Z_COLOR if show_height_graph else (80, 80, 80)
        pitch_color = PITCH_COLOR if show_pitch_graph else (80, 80, 80)
        roll_color = ROLL_COLOR if show_roll_graph else (80, 80, 80)
        
        cv2.putText(img, f"Height: {z_data[-1]:.2f}cm (S{SERVO_CONFIG['height']['index']}: {height_pwm})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, height_color, 2)
        cv2.putText(img, f"Pitch: {pitch_data[-1]:.1f}° (S{SERVO_CONFIG['pitch']['index']}: {pitch_pwm})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pitch_color, 2)
        cv2.putText(img, f"Roll: {roll_data[-1]:.1f}° (S{SERVO_CONFIG['roll']['index']}: {roll_pwm})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, roll_color, 2)
        
        time_window = WINDOW_TIME
        cv2.putText(img, f"Scroll: {SCROLL_SPEED}px/s | Window: {time_window:.1f}s | Points: {len(z_data)}/{MAX_POINTS}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # Draw scale labels
    cv2.putText(img, f"+{max(Z_MAX, PITCH_MAX, ROLL_MAX)}", (WINDOW_WIDTH - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(img, "0", (WINDOW_WIDTH - 20, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(img, f"{min(Z_MIN, PITCH_MIN, ROLL_MIN)}", (WINDOW_WIDTH - 70, WINDOW_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    # Replace the keyboard instructions line in draw_plot() with:
    
    # Recording status
    if recording_state == "RECORDING" and recording_start_time:
        duration = time.time() - recording_start_time
        cv2.putText(img, f"RECORDING: {duration:.1f}s | Data points: {len(recording_data)}", (WINDOW_WIDTH - 400, WINDOW_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RECORDING_COLOR, 2)

    # Instructions
    cv2.putText(img, f"Scroll Speed: {SCROLL_SPEED} pixels/second (adjust SCROLL_SPEED in code)", (10, WINDOW_HEIGHT - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(img, "Haptic control -> Robot servos via Serial | Press 'q' to quit | Click buttons to interact", (10, WINDOW_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(img, "Keyboard: SPACE=record toggle | S=serial toggle | C=clear graph | Mouse=GUI controls", (10, WINDOW_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # Serial status in plot area
    serial_status_text = f"Serial: {'ON' if serial_enabled else 'OFF'}"
    serial_status_color = (0, 255, 0) if serial_enabled else (0, 0, 255)
    cv2.putText(img, serial_status_text, (WINDOW_WIDTH - 200, WINDOW_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, serial_status_color, 2)

# Vector math for line constraints (same as original)
def normalize_vector(v):
    length = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
    if length < 1e-6:
        return [0.0, 0.0, 0.0]
    return [v[0]/length, v[1]/length, v[2]/length]

def dot_product(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vector_subtract(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def vector_scale(v, scale):
    return [v[0]*scale, v[1]*scale, v[2]*scale]

def vector_add(a, b):
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def calculate_line_constraint_force(current_pos, line_point, line_direction):
    line_dir = normalize_vector(line_direction)
    to_current = vector_subtract(current_pos, line_point)
    projection_length = dot_product(to_current, line_dir)
    projection_point = vector_add(line_point, vector_scale(line_dir, projection_length))
    constraint_vector = vector_subtract(projection_point, current_pos)
    distance = (constraint_vector[0]**2 + constraint_vector[1]**2 + constraint_vector[2]**2)**0.5
    
    SPRING_CONSTANT = 200.0
    MAX_FORCE = 10.0
    force_magnitude = min(SPRING_CONSTANT * distance, MAX_FORCE)
    
    if distance > 1e-6:
        force_direction = [constraint_vector[0]/distance, constraint_vector[1]/distance, constraint_vector[2]/distance]
        force = vector_scale(force_direction, force_magnitude)
    else:
        force = [0.0, 0.0, 0.0]
    
    return force

def main():
    global ser, serial_enabled, recording_state, recording_data, playback_data

    if recording_state == "PREVIEWING" and frame_counter % 100 == 0 and DEBUG_MODE:  # Print every 100 frames when previewing
        print(f"DEBUG: Main loop - recording_state={recording_state}, len(playback_data)={len(playback_data) if playback_data else 0}")

    # Load saved animations at startup
    load_saved_animations()
    
    # Initialize label manager
    label_manager = LabelManager()
    
    # Initialize serial connection only if enabled
    ser = None
    if ENABLE_SERIAL_OUTPUT:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
            print(f"✅ Serial connection established on {SERIAL_PORT} at {SERIAL_BAUD} baud")
            time.sleep(2)  # Allow ESP to reset
        except Exception as e:
            print(f"⚠️  Failed to open serial port {SERIAL_PORT}: {e}")
            print("🔄 Continuing without serial output (can be toggled in GUI)")
            serial_enabled = False
    else:
        print("📴 Serial output disabled in configuration")
        serial_enabled = False
    
    # Initialize haptic device
    device_id = fdsdk.open()
    if device_id < 0:
        print("❌ Failed to open haptic device.")
        if ser:
            ser.close()
        return
    
    print(f"✅ Haptic device opened successfully. ID: {device_id}")
    
    # Setup device
    fdsdk.reset(device_id)
    time.sleep(0.5)
    fdsdk.enableForce(1, device_id)
    
    # Z-line constraint parameters
    LINE_POINT = [0.0, 0.0, 0.0]
    LINE_DIRECTION = [0.0, 0.0, 1.0]
    
    # Create OpenCV window
    cv2.namedWindow('Haptic Robot Animation Recorder', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Haptic Robot Animation Recorder', handle_mouse_click, label_manager)
    img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    
    print("\n=== Haptic Robot Animation Recorder ===")
    print(f"Height (Z) -> Servo {SERVO_CONFIG['height']['index']} | Range: {SERVO_CONFIG['height']['pwm_min']}-{SERVO_CONFIG['height']['pwm_max']} | Center: {SERVO_CONFIG['height']['pwm_center']}")
    print(f"Pitch (Joint 1) -> Servo {SERVO_CONFIG['pitch']['index']} | Range: {SERVO_CONFIG['pitch']['pwm_min']}-{SERVO_CONFIG['pitch']['pwm_max']} | Center: {SERVO_CONFIG['pitch']['pwm_center']}") 
    print(f"Roll (Joint 2) -> Servo {SERVO_CONFIG['roll']['index']} | Range: {SERVO_CONFIG['roll']['pwm_min']}-{SERVO_CONFIG['roll']['pwm_max']} | Center: {SERVO_CONFIG['roll']['pwm_center']}")
    print(f"🔌 Serial output: {'ENABLED' if serial_enabled else 'DISABLED'} (toggle with button)")
    print(f"📁 Recordings will be saved to: {RECORDINGS_DIR}/")
    print(f"🏷️  Available gestures: {', '.join(label_manager.gestures)}")
    print(f"🎭 Available personalities: {', '.join(label_manager.personalities)}")
    print("🖱️  Click buttons to start/stop recording and edit labels")
    print("Press 'q' to quit")
    
    frame_counter = 0
    data_collection_counter = 0
    last_data_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # Get position
            position = [0.0, 0.0, 0.0]
            pos_result = fdsdk.getPosition(position, device_id)
            
            # Get joint angles
            joint_angles = [0.0, 0.0, 0.0]
            angles_result = fdsdk.getOrientationDeg(joint_angles, device_id)
            
            if pos_result >= 0 and angles_result >= 0:
                # Apply haptic constraint force
                force = calculate_line_constraint_force(position, LINE_POINT, LINE_DIRECTION)
                fdsdk.setForce(force, device_id)
                
                # Extract channels
                height_cm = position[2] * 100  # Convert to cm
                pitch_deg = joint_angles[1]    # Joint 1 as pitch
                roll_deg = joint_angles[0]     # Joint 2 as roll
                
                # Apply scaling to movements (only when not previewing)
                if recording_state != "PREVIEWING":
                    height_cm, pitch_deg, roll_deg = apply_movement_scaling(height_cm, pitch_deg, roll_deg)
                
                # Convert to PWM values
                height_pwm = map_to_pwm(height_cm, Z_MIN, Z_MAX, SERVO_CONFIG['height'])
                pitch_pwm = map_to_pwm(pitch_deg, PITCH_MIN, PITCH_MAX, SERVO_CONFIG['pitch'])
                roll_pwm = map_to_pwm(roll_deg, ROLL_MIN, ROLL_MAX, SERVO_CONFIG['roll'])
                
                # Handle playback during PREVIEWING state (call every frame for smooth playback)
                if recording_state == "PREVIEWING":
                    if DEBUG_MODE:
                        print("DEBUG: About to call update_playback")  # ADD THIS LINE
                    playback_values = update_playback(ser)
                    if playback_values:
                        if DEBUG_MODE:
                            print(f"DEBUG: Got playback values: H:{playback_values['height_pwm']} P:{playback_values['pitch_pwm']} R:{playback_values['roll_pwm']}")  # ADD THIS LINE
                        height_cm = playback_values['height_cm']
                        pitch_deg = playback_values['pitch_deg']
                        roll_deg = playback_values['roll_deg']
                        height_pwm = playback_values['height_pwm']
                        pitch_pwm = playback_values['pitch_pwm']
                        roll_pwm = playback_values['roll_pwm']
                    else:
                        if DEBUG_MODE:
                            print("DEBUG: update_playback returned None")  # ADD THIS LINE
                
                # Collect data at specified rate (10 times per second)
                if current_time - last_data_time >= (1.0 / DATA_POINTS_PER_SECOND):
                    # Add smoothed data
                    height_smooth = add_smoothed_data(height_cm, z_data_raw, z_data)
                    pitch_smooth = add_smoothed_data(pitch_deg, pitch_data_raw, pitch_data)
                    roll_smooth = add_smoothed_data(roll_deg, roll_data_raw, roll_data)
                    
                    # Record data point if recording
                    if recording_state == "RECORDING":
                        record_data_point(height_cm, pitch_deg, roll_deg, height_pwm, pitch_pwm, roll_pwm)
                    
                    last_data_time = current_time
                    data_collection_counter += 1
                
                # In the main loop, replace the serial command section with:
                if frame_counter % 3 == 0 and recording_state != "PREVIEWING":
                    # Use smoothed values for serial output instead of raw values
                    if len(z_data) > 0 and len(pitch_data) > 0 and len(roll_data) > 0 and ENABLE_SMOOTHING:
                        smooth_height_pwm = map_to_pwm(z_data[-1], Z_MIN, Z_MAX, SERVO_CONFIG['height'])
                        smooth_pitch_pwm = map_to_pwm(pitch_data[-1], PITCH_MIN, PITCH_MAX, SERVO_CONFIG['pitch'])
                        smooth_roll_pwm = map_to_pwm(roll_data[-1], ROLL_MIN, ROLL_MAX, SERVO_CONFIG['roll'])
                        send_servo_commands(ser, smooth_height_pwm, smooth_pitch_pwm, smooth_roll_pwm)
                    else:
                        send_servo_commands(ser, height_pwm, pitch_pwm, roll_pwm)
                
                # Update display at specified FPS (20 FPS for smoother looking charts)
                if frame_counter % (100 // DISPLAY_FPS) == 0:
                    draw_plot(img)
                    draw_gui_controls(img, label_manager)
                    cv2.imshow('Haptic Robot Animation Recorder', img)
                    
                    # Print current values less frequently for cleaner output
                    if frame_counter % 20 == 0:  # Every 200ms
                        serial_status = "📡" if serial_enabled else "📴"
                        status = f"{serial_status} H: {height_cm:.1f}cm({height_pwm}) | P: {pitch_deg:.1f}°({pitch_pwm}) | R: {roll_deg:.1f}°({roll_pwm}) | Points: {len(z_data)}"
                        if recording_state == "RECORDING":
                            status += f" | 🔴 REC: {len(recording_data)} pts"
                        elif recording_state == "PREVIEWING":
                            status += f" | ⏯️  PREVIEW"
                        print(status)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Spacebar for record toggle
                        if recording_state == "IDLE":
                            start_recording(label_manager, current_gesture_index, current_personality_index)
                        elif recording_state == "RECORDING":
                            stop_recording(label_manager, current_gesture_index, current_personality_index)
                        elif recording_state == "PREVIEWING":
                            save_recording(label_manager, current_gesture_index, current_personality_index)
                    elif key == 27:  # ESC key
                        if recording_state in ["RECORDING", "PREVIEWING"]:
                            cancel_recording()
                        elif gui_mode != "NORMAL":
                            gui_mode = "NORMAL"
                            input_text = ""
                    elif key == 13:  # ENTER key
                        if recording_state == "PREVIEWING":
                            save_recording(label_manager, current_gesture_index, current_personality_index)
                    elif key == ord('r') or key == ord('R'):  # R key for re-record
                        if recording_state == "PREVIEWING":
                            recording_state = "RECORDING"
                            recording_data = []
                            recording_start_time = time.time()
                            print("🔄 Re-recording started")
                    elif key == ord('s'):  # 's' key to toggle serial (only when not previewing)
                        if recording_state != "PREVIEWING":
                            serial_enabled = not serial_enabled
                            status = "ENABLED" if serial_enabled else "DISABLED"
                            print(f"🔌 Serial output {status}")
                    elif key == ord('c') or key == ord('C'):  # C key to clear graph
                        clear_graph_data()
                    else:
                        handle_keyboard_input(key, label_manager)
                
                frame_counter += 1
            
            time.sleep(0.01)  # 100 FPS for haptic feedback
    
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")
    
    finally:
        # Stop recording if active
        if recording_state == "RECORDING":
            stop_recording(label_manager, current_gesture_index, current_personality_index)
        
        # Cleanup       
        fdsdk.setForce([0.0, 0.0, 0.0], device_id)
        fdsdk.enableForce(0, device_id)
        fdsdk.close(device_id)
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("🔌 Devices closed.")
        print(f"📂 Check {RECORDINGS_DIR}/ for your recorded animations")

if __name__ == "__main__":
    main()