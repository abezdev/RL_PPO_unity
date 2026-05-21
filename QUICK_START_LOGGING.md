# 🎯 Quick Start: Enable Detailed Logging

## 30-Second Setup

### **Want to see tensor shapes and pixel values?**

**Step 1**: Edit `main.py` - Find this line in `_main_game_loop()`:
```python
if self.current_state == "FARMING":
    self._handle_farming()
```

**Step 2**: Change to enable verbose for first 10 steps:
```python
if self.current_state == "FARMING":
    self._handle_farming(verbose=(self.step_count < 10))
```

**Step 3**: Run and watch the detailed logs!

---

## What You'll See

**First step output**:
```
======================================================================
STEP 1 - FARMING WORKFLOW
======================================================================

1️⃣  [CAPTURE] Screen grabbed
   Shape: (400, 1000, 3), dtype: uint8
   Range: [0.000, 255.000]

📷 [PREPROCESS] Starting screen preprocessing...
   Input shape: (400, 1000, 3), dtype: uint8
   Input range: [0.000, 255.000]
   ✓ Resized to (64, 64, 3), range: [0.000, 255.000]
   ✓ Converted to grayscale (64, 64, 1), range: [0.000, 255.000]
   ✓ Normalized to (64, 64, 1), range: [0.000, 1.000]
   ✓ Added to frame buffer (buffer size: 1/4)
   ✓ Stacked output shape: (64, 64, 4), range: [0.000, 0.500]

🧠 [ACTION_SELECT] Getting action and value...
   Input batch shape: (1, 64, 64, 4), range: [0.000, 0.500]
   Policy logits shape: (1, 5), range: [-0.234, 0.567]
   Value output shape: (1, 1), value: 0.1234
   Action probs: [0.18 0.22 0.25 0.20 0.15]
   Selected action: 2, prob: 0.25
   Log prob: -1.3863, Value: 0.1234

💰 [REWARD] Calculating reward...
   State: XP=0.0 (prev=0.0), HP=100.0 (prev=100.0)
   Action: 2, Rune detected: False
   📊 Final reward: 0.0000

8️⃣  [BUFFER] Transition stored
   Buffer size: 1/2048

   📊 Step 1 | Reward: 0.0000 | Episode Total: 0.00
```

---

## Test Without Running Game

### Run complete logging demo:
```powershell
cd C:\Users\abez\source\repos\unity_ppo1
.\myenv313\Scripts\python.exe test_logging.py
```

**Output**: Full workflow with 5 simulated steps, all logging visible!

---

## Logging At Different Points

### **Just preprocessing**:
```python
processed = agent.preprocess_screen(screen, verbose=True)
```

### **Just action selection**:
```python
action, log_prob, value = agent.get_action_and_value(screen, verbose=True)
```

### **Just rewards**:
```python
reward = calc.calculate_reward(state_data, action, rune_detected, verbose=True)
```

### **Everything (full step)**:
```python
self._handle_farming(verbose=True)
```

---

## Common Log Patterns to Check

| Log | What to look for | Problem if |
|-----|-----------------|-----------|
| `[PREPROCESS]` | Shape: (64, 64, 4) | Not (64, 64, 4) = scaling issue |
| `[PREPROCESS]` | Range: [0.0, 1.0] | > 1.0 = not normalized |
| `[ACTION_SELECT]` | Probs sum ≈ 1.0 | < 0.99 or > 1.01 = softmax broken |
| `[ACTION_SELECT]` | Value: reasonable | NaN or ±inf = network broken |
| `[REWARD]` | XP gain or HP loss | All 0 = OCR not working |
| `[BUFFER]` | Growing | Stuck at same number = training loop stuck |

---

## Enable/Disable Quickly

```python
# At the top of main.py
DEBUG_VERBOSE = False  # Change to True to enable

# In _handle_farming():
self._handle_farming(verbose=DEBUG_VERBOSE)
```

Then quickly toggle:
```python
DEBUG_VERBOSE = True  # Turn on
# Run a few steps...
DEBUG_VERBOSE = False  # Turn off
```

---

## That's It! 🚀

Now when you run, you'll see exactly what's happening at each step:
- ✅ Screen shapes and pixel ranges
- ✅ Network input/output shapes  
- ✅ Action probabilities
- ✅ Reward breakdown
- ✅ Buffer filling up
- ✅ Training progress

Happy debugging! 🎯
