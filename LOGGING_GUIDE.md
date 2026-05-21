# 🔍 Logging Guide for Workflow Debugging

## Quick Usage

To see detailed tensor shapes and pixel value ranges, run with verbose mode:

### Option 1: Single Step Verbose Mode
```python
# In main.py, modify _handle_farming() call:
if self.step_count == 1:  # Only first step
    self._handle_farming(verbose=True)
else:
    self._handle_farming(verbose=False)
```

### Option 2: Every N Steps
```python
# In _handle_farming():
if self.step_count % 10 == 0:  # Every 10 steps
    self._handle_farming(verbose=True)
```

### Option 3: Enable via Command Line
```powershell
# Add to main.py at the end:
import sys
VERBOSE = "--verbose" in sys.argv

# Then in BotManager.__init__():
self.verbose = VERBOSE

# Then in _handle_farming():
self._handle_farming(verbose=self.verbose)

# Run with:
python main.py --verbose
```

---

## 📊 What Each Log Shows

### **1. Screen Capture** (`get_current_screen`)
```
📷 [CAPTURE] Screen grabbed
   Shape: (400, 1000, 3), dtype: uint8
   Range: [0.000, 255.000]
```
- **Shape**: Original captured region dimensions
- **Range**: Min/max pixel values (0-255 for raw, 0-1 for normalized)

### **2. Preprocessing** (`preprocess_screen`)
```
📷 [PREPROCESS] Starting screen preprocessing...
   Input shape: (400, 1000, 3), dtype: uint8
   Input range: [0.000, 255.000]
   ✓ Resized to (64, 64, 3), range: [0.000, 255.000]
   ✓ Converted to grayscale (64, 64, 1), range: [0.000, 255.000]
   ✓ Normalized to (64, 64, 1), range: [0.000, 1.000]
   ✓ Added to frame buffer (buffer size: 1/4)
   ✓ Stacked output shape: (64, 64, 4), range: [0.000, 0.667]
```
- **Resized**: Should be 64x64
- **Grayscale**: Single channel
- **Normalized**: Values 0.0-1.0
- **Frame buffer**: Growing from 1/4 to 4/4 initially
- **Stacked**: Full (64, 64, 4) tensor when buffer is full

### **3. Action Selection** (`get_action_and_value`)
```
🧠 [ACTION_SELECT] Getting action and value...
   📷 [PREPROCESS] Starting screen preprocessing...
   ...
   Input batch shape: (1, 64, 64, 4), range: [0.000, 1.000]
   Policy logits shape: (1, 5), range: [-1.234, 2.456]
   Value output shape: (1, 1), value: 0.3456
   Action probs: [0.12 0.34 0.28 0.18 0.08]
   Selected action: 1, prob: 0.34
   Log prob: -1.0788, Value: 0.3456
```
- **Batch shape**: Should be (1, 64, 64, 4) - adds batch dimension
- **Logits range**: Can be any value, typically [-5, 5]
- **Value**: Scalar estimate of state value
- **Probs**: Should sum to ~1.0 (probabilities)
- **Log prob**: Should be negative (log of probability < 1)

### **4. Reward Calculation** (`calculate_reward`)
```
💰 [REWARD] Calculating reward...
   State: XP=150.0 (prev=100.0), HP=90.0 (prev=100.0)
   Action: 1, Rune detected: False
   ✓ XP gain: +5.0000 (from 50.0 xp)
   ✗ HP loss: -0.5000 (lost 10.0 hp)
   📊 Final reward: 4.5000
```
- **State**: Current vs previous game metrics
- **Reward components**: Individual penalties/bonuses
- **Final reward**: Sum of all components

### **5. Buffer Status** (`store_transition`)
```
8️⃣  [BUFFER] Transition stored
   Buffer size: 45/2048
```
- **Buffer size**: Shows trajectory collection progress
- When buffer hits 128, training starts

### **6. Training** (when buffer fills)
```
📚 Training PPO on collected trajectories (128 steps)...
   ✅ Training completed. Policy Loss: 0.1234, Value Loss: 0.5678
   Episode Reward: 45.23
```
- **Policy Loss**: Should decrease over time
- **Value Loss**: Should decrease over time
- **Episode Reward**: Running total

---

## 🎯 Troubleshooting with Logs

### **Problem: Wrong tensor shapes**
- Check [PREPROCESS] output shape
- Should be (64, 64, 4) after frame stacking

### **Problem: Black/white images**
- Check [PREPROCESS] range
- If range is [0, 255], grayscale conversion failed
- If range is [0, 1], normalization worked

### **Problem: NaN in networks**
- Check logits range in [ACTION_SELECT]
- If range is [-inf, inf], network is broken
- Check value output for NaN

### **Problem: Rewards all zero**
- Check [REWARD] calculation
- Verify XP/HP extraction is working
- Check reward component values

### **Problem: Action prob distribution odd**
- Check if probs sum to ~1.0
- If not, softmax is failing
- Check logits range first

---

## 📝 Adding Custom Logs

Template for adding logs:
```python
if verbose:
    print(f"🔹 [STEP_NAME] Description...")
    print(f"   {variable_name}: {variable_value}")
    print(f"   Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    print(f"   Shape: {tensor.shape}")
```

Key metrics to log:
1. **Shape**: `variable.shape`
2. **Data type**: `variable.dtype`
3. **Range**: `[variable.min(), variable.max()]`
4. **Mean/std**: `[variable.mean(), variable.std()]`

---

## 🚀 Enable Logging in Real-Time

Add to `main.py` at the start:
```python
# Enable verbose logging for debugging
VERBOSE_LOGGING = False  # Change to True for detailed logs

# In BotManager._main_game_loop():
if VERBOSE_LOGGING and self.step_count <= 5:
    self._handle_farming(verbose=True)
else:
    self._handle_farming(verbose=False)
```

Then toggle quickly:
```python
VERBOSE_LOGGING = True  # Turn on
# ... run a few steps ...
VERBOSE_LOGGING = False  # Turn off
```
