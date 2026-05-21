import pydirectinput
import time
import threading
import os
import cv2
import numpy as np
import mss
import re

from typing import Literal
from datetime import datetime
from collections import deque
#C:\Users\abez\Downloads\Xpytesseract_exe
#r"C:\Program Files\Tesseract-OCR\tesseract.exe" C:\Users\abez\AppData\Local\Programs\Tesseract-OCR
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\abez\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
except ImportError:
    print("⚠️  pytesseract not installed. Install with: pip install pytesseract")
    pytesseract = None

from rl_agent.ppomsodel import PPOAgent, RewardCalculator
from vision.cha_solvrs import CaptchaSolver

# Define the possible operational states for the bot
BotState = Literal["FARMING", "SOLVING_CAPTCHA"]

class BotManager:
    """
    Manages the core game loop, state transitions, and interaction between
    the RL agent and the vision solver.
    Implements trajectory collection and periodic PPO training.
    """
    def __init__(self, screen_shape: tuple[int, int, int] = (400, 1000, 3)):
        print("🤖 Initializing BotManager...")
        # Initialize RL agent
        self.rl_agent = PPOAgent(input_shape=screen_shape, action_space_size=5)
        self.captcha_solver = CaptchaSolver()
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Bot state management
        self.current_state: BotState = "FARMING"
        self.running = False
        
        # Training parameters
        self.step_count = 0
        self.episode_reward = 0.0
        self.training_interval = 128  # Train after collecting 128 steps
        self.max_episode_steps = 1000
        
        # Key state tracking for continuous key holds
        self.pressed_keys = []

    def _screen_watcher_loop(self):
        """
        Continuously watches the screen for state-changing events (like a Captcha appearing).
        This runs on a separate thread.
        """
        print("👁️ Starting screen-watching thread...")
        while self.running:
            # --- PLACEHOLDER: Continuous Screen-Watching Logic ---
            # In a real implementation, this would capture the screen and analyze it.
            
            # Simulated state transition logic:
            # Check if a Captcha is detected *only* if currently farming
            if self.current_state == "FARMING":
                # placeholder_captcha_detected = self.captcha_solver.detect_captcha(get_current_screen()) 
                placeholder_captcha_detected = False # Replace with real detection
                
                if placeholder_captcha_detected:
                    print("⚠️ Captcha detected! Initiating state transition.")
                    self.current_state = "SOLVING_CAPTCHA"
            # ---------------------------------------------------
            
            time.sleep(110.5) # Throttle the screen-watching frequency

    def start_bot(self):
        """Starts the main operation and the screen-watching loop."""
        if self.running:
            print("Bot is already running.")
            return

        self.running = True
        # Start the screen-watching loop in a background thread
        watcher_thread = threading.Thread(target=self._screen_watcher_loop, daemon=True)
        watcher_thread.start()
        
        print(f"🚀 Bot started. Initial state: {self.current_state}")
        self._main_game_loop()

    def stop_bot(self):
        """Stops the bot."""
        self.running = False
        print("🛑 Bot stopping...")

    def _main_game_loop(self):
        """
        The main execution loop that switches behavior based on the current state.
        """
        while self.running:
            # --- State Transition Logic ---
            if self.current_state == "FARMING":
                self._handle_farming(True)
            elif self.current_state == "SOLVING_CAPTCHA":
                self._handle_captcha_solve()
            # ------------------------------
            
            time.sleep(0.81)  # Main loop throttle

    def _handle_farming(self, verbose: bool = False):
        """
        Executes actions for the FARMING state using the RL agent with detailed logging.
        Set verbose=True to see tensor shapes and pixel ranges at each step.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP {self.step_count} - FARMING WORKFLOW")
            print(f"{'='*70}")
        
        # 1. Capture screen
        current_screen = get_current_screen()
        if current_screen is None:
            print(f"❌ Screen capture failed")
            return
        
        if verbose:
            print(f"\n1️⃣  [CAPTURE] Screen grabbed")
            print(f"   Shape: {current_screen.shape}, dtype: {current_screen.dtype}")
            print(f"   Range: [{current_screen.min():.3f}, {current_screen.max():.3f}]")
        
        # 2. Get RL agent action (with logging)
        action, log_prob, value = self.rl_agent.get_action_and_value(current_screen, verbose=verbose)
        
        if not verbose:
            print(f"[Step {self.step_count:3d}] Action: {action} | Value: {value:7.3f} | LogProb: {log_prob:8.4f}")
        
        # 3. Execute the action in-game
        execute_game_action(action, self)
        
        # 4. Extract game state
        state_data = {
            'xp': extract_xp_from_screen(current_screen),
            'hp': extract_hp_from_screen(current_screen) if False else 100.0,
        }
        
        # 5. Check for runes
        rune_detected = False
        
        # 6. Calculate reward with logging
        reward = self.reward_calculator.calculate_reward(state_data, action, rune_detected, verbose=verbose)
        self.episode_reward += reward
        
        # 7. Episode done?
        done = self.step_count >= self.max_episode_steps
        
        # 8. Store transition
        self.rl_agent.store_transition(
            state=current_screen,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done
        )
        
        if verbose:
            print(f"\n8️⃣  [BUFFER] Transition stored")
            print(f"   Buffer size: {len(self.rl_agent.trajectory_buffer['states'])}/2048")

        # 9. Increment step counter
        self.step_count += 1
        
        if verbose or self.step_count % 10 == 0:
            print(f"   📊 Step {self.step_count} | Reward: {reward:7.4f} | Episode Total: {self.episode_reward:9.2f}")
        
        
        # 10. Periodically train on collected trajectories
        if self.step_count % self.training_interval == 0:
            print(f"\n📚 Training PPO on collected trajectories ({self.step_count} steps)...")
            self.rl_agent.train(epochs=3)
            print(f"   Episode Reward: {self.episode_reward:.2f}\n")
            self.step_count = 0
            self.episode_reward = 0.0
        
        # Small delay to allow game response
        time.sleep(0.05)

    def _handle_captcha_solve(self):
        """Executes actions for the SOLVING_CAPTCHA state using the vision solver."""
        print(f"[{time.time():.2f}] 🔒 State: SOLVING_CAPTCHA. Captcha Solver active...")
        
        try:
            # 1. Detect the rune (placeholder)
            # self.captcha_solver.detect_rune(get_current_screen())
            
            # 2. Crop the arrows/sequence area (placeholder)
            # self.captcha_solver.crop_arrows(get_current_screen())
            
            # 3. Solve and execute the sequence
            # success = self.captcha_solver.solve_and_execute_sequence(cropped_image)
            
            # --- PLACEHOLDER: Solve and Execute ---
            print("Attempting to solve the captcha...")
            time.sleep(3) # Simulate solving time
            success = True # Assume success for the skeleton
            # ------------------------------------

            if success:
                print("✅ Captcha solved successfully. Returning to FARMING.")
                # Transition back to the Farming state
                self.current_state = "FARMING"
            else:
                print("❌ Captcha solving failed. Retrying or handling error...")
                # Implement retry or exit logic here
                time.sleep(5)
                
        except Exception as e:
            print(f"An error occurred during captcha solving: {e}")
            # Transition to a safe state or stop
            self.stop_bot()

# ===== SCREEN CAPTURE & PREPROCESSING =====
def get_current_screen(region=None, target_size=(256, 256)):
        """
        Captures a specific region of the screen, preprocesses it, and returns 
        a NumPy array ready for a CNN.

        Args:
            region (dict): A dictionary defining the bounding box for the capture.
                        Example: {'top': 100, 'left': 100, 'width': 500, 'height': 400}
            target_size (tuple): The desired (height, width) for the CNN input.

        Returns:
            np.ndarray: The preprocessed image as a NumPy array (H, W, C).
                        Returns None if capture fails.
        """

        region = {
            'top': 200,
            'left': 2300,
            'width': 1000,
            'height': 400
        }

        output_folder = "C:\\Users\\abez\\source\\repos\\unity_ppo1\\images"#"C:\\Users\\abez\\source\\repos\\unity_ppo1\\images"
        try:
            # Create the output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # 1. Capture the Specific Region using MSS
            with mss.mss() as sct:
                sct_img = sct.grab(region)
                
                # Convert the raw bytes to a NumPy array
                img_array = np.array(sct_img)
                
                # The MSS array is (H, W, 4) in BGRA format.
                # Convert it to BGR (H, W, 3), which is the standard format for saving 
                # image files (like .png or .jpg) using OpenCV.
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                


                #Define File Path and Save
                
                # Create a unique filename using the current timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}_initial_image.png"
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, img_bgr)#qq img_bgr/img_array

                print("Start Screen Capture ~ :", os.path.exists(output_path))
                print("Path:", output_path)

                # 2. Preprocess the Image
            
                # Resize the image to the required CNN input dimensions
                resized_img = cv2.resize(img_bgr, target_size)

                print(img_array.shape) #(400, 1000, 3)
                print(resized_img.shape) #(256, 256, 3)
                print("~~~~~~~~resized_img~~~~~~~~~~~~~")

                filename = f"screenshot_{timestamp}_resized into target-256.png"
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, resized_img)


                # Normalize the pixel values from [0, 255] to [0.0, 1.0]
                normalized_img = resized_img.astype('float32') / 255.0
                


                # 3. Return the Processed Array
                return normalized_img






        except Exception as e:
            print(f"An error occurred during screen capture or preprocessing: {e}")
            return None


# Map integer actions to keys
ACTION_MAP = {
    0: 'space',      # jump
    1: 'd',          # Turn/Move Right
    2: 'a',          # Turn/Move Left
    3: 'z',          # Attack
    4: None          # Wait / No-op
}
def execute_game_action(action_idx: int, bot_manager: 'BotManager' = None):
    """
    Translates the RL agent's integer output into a game key press.
    Uses key state tracking to hold keys down for continuous movement.
    """
    key = ACTION_MAP.get(action_idx)
    
    if bot_manager is None:
        # Fallback: simple press without tracking
        if key is not None:
            pydirectinput.press(key)
            print(f"👉 Executed Action {action_idx}: Pressed '{key}'")
        else:
            time.sleep(0.05)  # Wait action
        return
    
    # Release keys not in current action
    keys_to_release = [k for k in bot_manager.pressed_keys if k != key]
    for k in keys_to_release:
        try:
            pydirectinput.keyUp(k)
            bot_manager.pressed_keys.remove(k)
        except:
            pass
    
    if key is None:
        # Action: Wait
        time.sleep(0.05)
        return
    
    # Press or hold the key
    if key not in bot_manager.pressed_keys:
        pydirectinput.keyDown(key)
        bot_manager.pressed_keys.append(key)
        print(f"👉 Action {action_idx}: Holding '{key}'")
    else:
        print(f"👉 Action {action_idx}: Continuing '{key}'")


# ===== GAME STATE EXTRACTION (OCR & Detection Implementation) =====
def extract_xp_from_screen(screen: np.ndarray, xp_region=(25, 20, 150, 80)) -> float:
    """
    Extracts XP value from the screen using OCR (similar to MapleAI Trainer).
    #(467, 760, 90, 15)  $() 
    Args:
        screen: Full screen numpy array
        xp_region: (x, y, width, height) tuple for XP display location
                   Adjust based on your game's UI layout
    
    Returns:
        Float representing current XP gained in this step
    """
    print("----pytesseract extract_xp_from_screen    extract_xp_from_screen  .")


    if pytesseract is None:
        print("pytesseract not available for XP extraction.")
        return 0.0
    
    try:
        x, y, w, h = xp_region
        
        # Ensure we're working with the original captured screen (not the 64x64 preprocessed one)
        # If screen is 64x64, return 0 (preprocessing already happened)
        #if screen.shape[0] == 64 and screen.shape[1] == 64:
        #    return 0.0
        
        # Crop the XP display area
        xp_frame = screen[y:y+h, x:x+w].copy()
        print("--1--")
        # Convert to 8-bit if needed
        if xp_frame.dtype == np.float32:
            xp_frame = (xp_frame * 255).astype(np.uint8)
        print("--2--")


              # Create a unique filename using the current timestamp
        output_folder = "C:\\Users\\abez\\source\\repos\\unity_ppo1\\images"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}_thisthe_expReward.png"
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, xp_frame)


        # Convert to grayscale
        if len(xp_frame.shape) == 3:
            xp_frame = cv2.cvtColor(xp_frame, cv2.COLOR_RGB2GRAY)
        print("--3--")
        # Posterize: threshold to binary for better OCR
        xp_frame[xp_frame >= 128] = 255
        xp_frame[xp_frame < 128] = 0




        filename = f"screenshot_{timestamp}_thisthe_expReward_after posturize.png"
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, xp_frame)
        """If you find the OCR is failing, you might want to try Otsu’s Binarization, 
        which automatically calculates the best threshold value based on the image's lighting:

        _, xp_frame = cv2.threshold(xp_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        """
        
        
        print("--4--")


        print("--5--")

        # OCR recognition
        xp_text = pytesseract.image_to_string(xp_frame).strip()
        print(f"📊 XP OCR Capture:@@@@@@@@@@@@@@@@@---@@@@@@@@@@@@@@@@@@@@@@>> {xp_text}")
        print("--6--")
        # Parse the result (typically formatted as "1234 98.16%")


        #xp_value = xp_text.split()[0] if xp_text else "0"
        print("--7--")
        # Sanitize OCR errors
        # xp_value = xp_value.replace("$", "5")
        # xp_value = xp_value.replace("§", "5")
        # xp_value = xp_value.replace("S", "5")
        # xp_value = xp_value.replace("O", "0")
        xp_text = xp_text.replace("$", "5").replace("§", "5").replace("S", "5").replace("O", "0")

        print("--8--")

        print("xp_text:", xp_text)
        xp_value = re.search(r"coins:\s*(\d+)", xp_text, re.IGNORECASE)
        print("xp_value@@@@@@@@@@@@@@@@@---@@@@@@@@@@@@@@@@@@@@@@>>", xp_value)

        xp_gained = int(xp_value)
        print(f"✅ Current XP:@@@@@@@@@@@@@@@@@---@@@@@@@@@@@@@@@@@@@@@@>> {xp_gained}")
        return float(xp_gained)
        
    except Exception as e:
        print(f"⚠️ Error extracting XP: {e}")
        return 0.0


def extract_hp_from_screen(screen: np.ndarray, hp_region=(239, 760, 80, 15)) -> float:
    """
    Extracts HP (Health Points) value from the screen using OCR.
    
    Args:
        screen: Full screen numpy array
        hp_region: (x, y, width, height) tuple for HP display location
                   Adjust based on your game's UI layout
    
    Returns:
        Float representing current HP percentage or absolute value
    """
    if pytesseract is None:
        return 100.0
    
    try:
        x, y, w, h = hp_region
        
        # If screen is 64x64 (preprocessed), return default
        if screen.shape[0] == 64 and screen.shape[1] == 64:
            return 100.0
        
        # Crop the HP display area
        hp_frame = screen[y:y+h, x:x+w].copy()
        
        # Convert to 8-bit if needed
        if hp_frame.dtype == np.float32:
            hp_frame = (hp_frame * 255).astype(np.uint8)
        
        # Convert to grayscale
        if len(hp_frame.shape) == 3:
            hp_frame = cv2.cvtColor(hp_frame, cv2.COLOR_RGB2GRAY)
        
        # Posterize: threshold to binary
        hp_frame[hp_frame >= 128] = 255
        hp_frame[hp_frame < 128] = 0
        
        # OCR recognition (typically "300/500")
        hp_text = pytesseract.image_to_string(hp_frame).strip()
        print(f"💚 HP OCR Capture: {hp_text}")
        
        # Parse HP value (take first number before "/")
        hp_value = hp_text.split("/")[0] if "/" in hp_text else hp_text.split()[0]
        
        # Sanitize OCR errors
        hp_value = hp_value.replace("$", "5")
        hp_value = hp_value.replace("§", "5")
        hp_value = hp_value.replace("S", "5")
        hp_value = hp_value.replace("O", "0")
        
        hp_current = int(hp_value)
        print(f"✅ Current HP: {hp_current}")
        return float(hp_current)
        
    except Exception as e:
        print(f"⚠️ Error extracting HP: {e}")
        return 100.0


if __name__ == "__main__":
    manager = BotManager()
    
    time.sleep(5.5)  # Give user time to switch to game window
    print("Program continues after the delay.")

    # In a real scenario, this would be wrapped in error handling and graceful shutdown
    try:
        manager.start_bot()
        # Keep the main thread alive while background threads run
        while manager.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupt received.")
    finally:
        manager.stop_bot()
        print("Application shut down.")