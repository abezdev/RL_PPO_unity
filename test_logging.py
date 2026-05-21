#!/usr/bin/env python3
"""
Test script to demonstrate logging at each step of the workflow.
Run with: python test_logging.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from rl_agent.ppomsodel import PPOAgent, RewardCalculator
import cv2

def test_preprocessing_pipeline():
    """Test and show preprocessing steps with logging."""
    print("\n" + "="*70)
    print("🧪 PREPROCESSING PIPELINE TEST")
    print("="*70)
    
    # Create PPO agent
    agent = PPOAgent(input_shape=(256, 256, 3), action_space_size=5, frame_stack_size=4)
    
    # Simulate raw screen captures (multiple frames)
    print("\n📸 Simulating 5 screen captures...")
    for frame_num in range(5):
        # Create random screen image
        raw_screen = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        print(f"\n--- Frame {frame_num+1} ---")
        print(f"Raw input: shape={raw_screen.shape}, range=[{raw_screen.min()}, {raw_screen.max()}]")
        
        # Preprocess with verbose logging
        stacked = agent.preprocess_screen(raw_screen, verbose=True)
        
        print(f"\nBuffer state: {len(agent.frame_buffer)}/{agent.frame_stack_size} frames")

def test_action_selection():
    """Test and show action selection with logging."""
    print("\n" + "="*70)
    print("🧠 ACTION SELECTION TEST")
    print("="*70)
    
    agent = PPOAgent(input_shape=(256, 256, 3), action_space_size=5, frame_stack_size=4)
    
    # Warm up the frame buffer
    print("\nWarming up frame buffer...")
    for i in range(4):
        raw_screen = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        agent.preprocess_screen(raw_screen, verbose=False)
    
    # Now get action with logging
    print("\n--- Getting Action (Frame 5) ---")
    raw_screen = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    action, log_prob, value = agent.get_action_and_value(raw_screen, verbose=True)

def test_reward_calculation():
    """Test and show reward calculation with logging."""
    print("\n" + "="*70)
    print("💰 REWARD CALCULATION TEST")
    print("="*70)
    
    calc = RewardCalculator()
    
    # Simulate game states
    states = [
        {'xp': 0, 'hp': 100},      # Initial
        {'xp': 50, 'hp': 95},      # Gained XP, took damage
        {'xp': 100, 'hp': 100},    # More XP, healed
        {'xp': 100, 'hp': 70},     # Took lots of damage
    ]
    
    for step, state in enumerate(states):
        print(f"\n--- Step {step+1} ---")
        reward = calc.calculate_reward(state, action_taken=1, rune_detected=False, verbose=True)
        print(f"→ Reward: {reward:.4f}")

def test_full_workflow():
    """Test the complete workflow from screen to reward."""
    print("\n" + "="*70)
    print("🔄 FULL WORKFLOW TEST (3 steps)")
    print("="*70)
    
    agent = PPOAgent(input_shape=(256, 256, 3), action_space_size=5, frame_stack_size=4)
    calc = RewardCalculator()
    
    for step in range(3):
        print(f"\n{'='*70}")
        print(f"STEP {step+1}")
        print(f"{'='*70}")
        
        # 1. Screen capture (simulated)
        raw_screen = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        print(f"\n1️⃣  Screen captured: {raw_screen.shape}")
        
        # 2. Preprocess
        processed = agent.preprocess_screen(raw_screen, verbose=True)
        
        # 3. Get action
        action, log_prob, value = agent.get_action_and_value(raw_screen, verbose=True)
        
        # 4. Simulate game state change
        state_data = {'xp': 50 * step, 'hp': 100 - 5*step}
        
        # 5. Calculate reward
        reward = calc.calculate_reward(state_data, action, False, verbose=True)
        
        # 6. Store trajectory
        agent.store_transition(raw_screen, action, reward, value, log_prob, done=False)
        print(f"\n8️⃣  Stored to buffer: {len(agent.trajectory_buffer['states'])}/2048")

if __name__ == "__main__":
    print("\n" + "🔍 LOGGING DEMONSTRATION SCRIPT")
    print("  Shows tensor shapes and pixel values at each step")
    
    # Run tests
    test_preprocessing_pipeline()
    test_action_selection()
    test_reward_calculation()
    test_full_workflow()
    
    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("  See above for detailed logging of tensor shapes and value ranges")
    print("="*70)
