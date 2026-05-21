import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque


class PPOAgent:
    """
    A reinforcement learning agent using the PPO algorithm to decide actions
    during the FARMING state based on screen input.
    
    Implements:
    - CNN-based Policy and Value networks
    - Image preprocessing (64x64 normalization)
    - Frame stacking for temporal information
    - Action selection from 5 discrete actions
    - PPO training with GAE (Generalized Advantage Estimation)
    """
    def __init__(self, input_shape: tuple = (256, 256, 3), action_space_size: int = 5, frame_stack_size: int = 4):
        """
        Args:
            input_shape: Original screen shape (height, width, channels)
            action_space_size: Number of discrete actions (default: 5)
            frame_stack_size: Number of frames to stack (default: 4)
        """
        self.input_shape = input_shape
        self.action_space_size = action_space_size
        self.frame_stack_size = frame_stack_size
        self.processed_shape = (64, 64, 1)  # Target preprocessed shape per frame
        self.stacked_shape = (64, 64, frame_stack_size)  # Stacked observation shape
        
        # Frame stacking buffer
        self.frame_buffer = deque(maxlen=frame_stack_size)
        
        # Hyperparameters
        self.learning_rate = 3e-4
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.clip_ratio = 0.2  # PPO clip ratio
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        
        # Build networks
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()
        
        # Optimizers
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Memory for trajectory collection
        self.trajectory_buffer = {
            'states': deque(maxlen=2048),
            'actions': deque(maxlen=2048),
            'rewards': deque(maxlen=2048),
            'values': deque(maxlen=2048),
            'log_probs': deque(maxlen=2048),
            'dones': deque(maxlen=2048),
        }
        
        print("🧠 PPOAgent initialized with CNN policy and value networks.")
        print(f"   Input shape: {self.input_shape} → Processed: {self.processed_shape}")
        print(f"   Action space: {self.action_space_size} actions")

    def _build_policy_network(self) -> keras.Model:
        """
        Builds the policy network: CNN → FC layers → output logits (5 actions).
        Input: Stacked frames (64, 64, frame_stack_size)
        """
        inputs = keras.Input(shape=self.stacked_shape, name="state_input")
        
        # CNN backbone
        x = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(inputs)
        x = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(x)
        x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
        x = layers.Flatten()(x)
        
        # FC layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # Output: action logits (5 actions)
        action_logits = layers.Dense(self.action_space_size, activation=None, name="action_logits")(x)
        
        model = keras.Model(inputs=inputs, outputs=action_logits)
        return model

    def _build_value_network(self) -> keras.Model:
        """
        Builds the value network: CNN → FC layers → single scalar value estimate.
        Input: Stacked frames (64, 64, frame_stack_size)
        """
        inputs = keras.Input(shape=self.stacked_shape, name="state_input")
        
        # CNN backbone (shared architecture)
        x = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(inputs)
        x = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(x)
        x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
        x = layers.Flatten()(x)
        
        # FC layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # Output: scalar value
        value = layers.Dense(1, activation=None, name="value")(x)
        
        model = keras.Model(inputs=inputs, outputs=value)
        return model

    def preprocess_screen(self, screen_input: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Preprocesses screen input with detailed logging.
        1. Resize to 64x64
        2. Convert to grayscale
        3. Normalize to [0, 1]
        4. Add to frame stack
        5. Return stacked frames
        """
        if verbose:
            print("\n📷 [PREPROCESS] Starting screen preprocessing...")
            if screen_input is not None:
                print(f"   Input shape: {screen_input.shape}, dtype: {screen_input.dtype}")
                print(f"   Input range: [{screen_input.min():.3f}, {screen_input.max():.3f}]")
        
        if screen_input is None or screen_input.size == 0:
            dummy_frame = np.zeros((64, 64, 1), dtype=np.float32)
            self.frame_buffer.append(dummy_frame)
            if verbose:
                print("   ⚠️  Empty input, using zeros")
            return self._get_stacked_frames(verbose=verbose)
        
        if not isinstance(screen_input, np.ndarray):
            screen_input = np.array(screen_input)
        
        # Resize to 64x64
        import cv2
        resized = cv2.resize(screen_input, (64, 64), interpolation=cv2.INTER_LINEAR)
        if verbose:
            print(f"   ✓ Resized to {resized.shape}, range: [{resized.min():.3f}, {resized.max():.3f}]")
        
        # Convert to grayscale if RGB
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            gray = np.expand_dims(gray, axis=-1)
        elif len(resized.shape) == 2:
            gray = np.expand_dims(resized, axis=-1)
        else:
            gray = resized
        
        if verbose:
            print(f"   ✓ Grayscale {gray.shape}, range: [{gray.min():.3f}, {gray.max():.3f}]")
        
        # Normalize to [0, 1]
        if gray.max() > 1.0:
            processed = gray.astype(np.float32) / 255.0
        else:
            processed = gray
        
        if verbose:
            print(f"   ✓ Normalized {processed.shape}, range: [{processed.min():.3f}, {processed.max():.3f}]")
        
        self.frame_buffer.append(processed)
        if verbose:
            print(f"   ✓ Frame buffer: {len(self.frame_buffer)}/{self.frame_stack_size}")
        
        stacked = self._get_stacked_frames(verbose=verbose)
        if verbose:
            print(f"   ✓ Final stacked: {stacked.shape}, range: [{stacked.min():.3f}, {stacked.max():.3f}]")
        
        return stacked
    
    def _get_stacked_frames(self, verbose: bool = True) -> np.ndarray:
        """
        Returns stacked frames from the buffer with logging.
        Pads with zeros if buffer is not full yet.
        
        Returns:
            Stacked tensor (64, 64, frame_stack_size)
        """
        stacked = np.zeros(self.stacked_shape, dtype=np.float32)
        
        # Fill with available frames
        for i, frame in enumerate(list(self.frame_buffer)):
            stacked[:, :, i] = frame[:, :, 0]
        # This converts the deque/list directly into a single block of memory
        # stacked = np.concatenate(list(self.frame_buffer), axis=-1)   
        
        if verbose and len(self.frame_buffer) == self.frame_stack_size:
            print(f"   📦 [STACK] Full buffer ready: {stacked.shape}")
        
        return stacked
    
    def reset_frame_buffer(self):
        """Reset the frame buffer (e.g., at episode start)."""
        self.frame_buffer.clear()

    def get_action(self, screen_input: np.ndarray, training: bool = False) -> int:
        """
        Given a screen frame, preprocess it and return an action (0-4).
        
        Args:
            screen_input: Raw screen image
            training: If True, sample from policy; if False, use greedy (argmax)
            
        Returns:
            Integer action (0-4)
        """
        # Preprocess
        processed = self.preprocess_screen(screen_input)
        processed_batch = np.expand_dims(processed, axis=0)  # Add batch dim
        
        # Get action logits
        logits = self.policy_network(processed_batch, training=False)
        logits = logits[0]  # Remove batch dim
        
        # Convert logits to probabilities
        probs = tf.nn.softmax(logits).numpy()
        
        if training:
            # Sample from policy during training
            action = np.random.choice(self.action_space_size, p=probs)
        else:
            # Greedy action during inference
            action = np.argmax(probs)
        
        return int(action)

    def get_action_and_value(self, screen_input: np.ndarray, verbose: bool = False):
        """
        Returns action, log probability, and value estimate with logging.
        Used during trajectory collection for training.
        
        Args:
            screen_input: Raw screen image
            verbose: Print detailed logging
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        if verbose:
            print("\n🧠 [ACTION_SELECT] Getting action and value...")
        
        # Preprocess
        processed = self.preprocess_screen(screen_input, verbose=verbose)
        processed_batch = np.expand_dims(processed, axis=0)
        
        if verbose:
            print(f"   Input batch shape: {processed_batch.shape}, range: [{processed_batch.min():.3f}, {processed_batch.max():.3f}]")
        
        # Get logits and value
        logits = self.policy_network(processed_batch, training=True)
        value = self.value_network(processed_batch, training=True)
        
        if verbose:
            print(f"   Policy logits shape: {logits.shape}, range: [{logits.numpy().min():.3f}, {logits.numpy().max():.3f}]")
            print(f"   Value output shape: {value.shape}, value: {value.numpy().flatten()[0]:.4f}")
        
        logits = logits[0]
        value = value[0, 0].numpy()
        
        # Sample action
        probs = tf.nn.softmax(logits).numpy()
        action = np.random.choice(self.action_space_size, p=probs)
        
        if verbose:
            print(f"   Action probs: {probs}")
            print(f"   Selected action: {action}, prob: {probs[action]:.4f}")
        
        # Compute log probability
        log_prob = np.log(probs[action] + 1e-10)
        
        if verbose:
            print(f"   Log prob: {log_prob:.4f}, Value: {value:.4f}")
        
        return int(action), float(log_prob), float(value)


    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store a transition in the replay buffer."""
        self.trajectory_buffer['states'].append(state)
        self.trajectory_buffer['actions'].append(action)
        self.trajectory_buffer['rewards'].append(reward)
        self.trajectory_buffer['values'].append(value)
        self.trajectory_buffer['log_probs'].append(log_prob)
        self.trajectory_buffer['dones'].append(done)

    def compute_advantages(self):
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        Returns advantages and returns for training.
        """
        rewards = list(self.trajectory_buffer['rewards'])
        values = list(self.trajectory_buffer['values'])
        dones = list(self.trajectory_buffer['dones'])
        
        advantages = []
        gae = 0
        
        # Compute GAE backward
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Assume terminal state
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(values)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns

    def train(self, epochs: int = 3):
        """
        Train on collected trajectory buffer using PPO loss.
        
        Args:
            epochs: Number of epochs to train on the buffer
        """
        if len(self.trajectory_buffer['states']) == 0:
            print("⚠️ No trajectories to train on.")
            return
        
        # Compute advantages
        advantages, returns = self.compute_advantages()
        
        # Convert buffer to arrays
        states = np.array([self.preprocess_screen(s) for s in self.trajectory_buffer['states']])
        actions = np.array(self.trajectory_buffer['actions'])
        old_log_probs = np.array(self.trajectory_buffer['log_probs'])
        
        # Training loop
        for epoch in range(epochs):
            with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
                # Forward pass
                logits = self.policy_network(states, training=True)
                values = self.value_network(states, training=True).squeeze()
                
                # Compute new log probabilities
                probs = tf.nn.softmax(logits)
                log_probs = tf.math.log(probs + 1e-10)
                selected_log_probs = tf.reduce_sum(
                    tf.one_hot(actions, self.action_space_size) * log_probs,
                    axis=1
                )
                
                # PPO loss
                ratio = tf.exp(selected_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                
                # Entropy bonus
                entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=1))
                total_policy_loss = policy_loss - self.entropy_coef * entropy
                
                # Value loss (MSE)
                value_loss = tf.reduce_mean(tf.square(values - returns))
            
            # Update policy network
            policy_grads = policy_tape.gradient(total_policy_loss, self.policy_network.trainable_weights)
            policy_grads, _ = tf.clip_by_global_norm(policy_grads, self.max_grad_norm)
            self.policy_optimizer.apply_gradients(
                zip(policy_grads, self.policy_network.trainable_weights)
            )
            
            # Update value network
            value_grads = value_tape.gradient(value_loss, self.value_network.trainable_weights)
            value_grads, _ = tf.clip_by_global_norm(value_grads, self.max_grad_norm)
            self.value_optimizer.apply_gradients(
                zip(value_grads, self.value_network.trainable_weights)
            )
        
        print(f"✅ Training completed. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        self.clear_buffer()

    def clear_buffer(self):
        """Clear the trajectory buffer after training."""
        for key in self.trajectory_buffer:
            self.trajectory_buffer[key].clear()

    def load(self, path: str):
        """Load trained models from disk."""
        try:
            self.policy_network = keras.models.load_model(f"{path}/policy_network")
            self.value_network = keras.models.load_model(f"{path}/value_network")
            print(f"✅ Models loaded from {path}")
        except Exception as e:
            print(f"❌ Failed to load models: {e}")

    def save(self, path: str):
        """Save trained models to disk."""
        try:
            self.policy_network.save(f"{path}/policy_network")
            self.value_network.save(f"{path}/value_network")
            print(f"✅ Models saved to {path}")
        except Exception as e:
            print(f"❌ Failed to save models: {e}")






#do i need this? or does ppo model handle reward calc?

class RewardCalculator:
    def __init__(self):
        self.previous_xp = 0
        self.previous_hp = 100
        
    def calculate_reward(self, current_state_data: dict, action_taken: int, rune_detected: bool, verbose: bool = False) -> float:
        """
        Calculates the reward for the current time step with logging.
        
        Args:
            current_state_data: Dict containing 'xp', 'hp', etc.
            action_taken: The integer action (0-4) the agent just took.
            rune_detected: Boolean, did the Vision system see a rune on screen?
            verbose: Print detailed logging
        """
        if verbose:
            print("\n💰 [REWARD] Calculating reward...")
        
        reward = 0.0
        
        current_xp = current_state_data.get('xp', 0)
        current_hp = current_state_data.get('hp', 100)
        
        if verbose:
            print(f"   State: XP={current_xp:.1f} (prev={self.previous_xp:.1f}), HP={current_hp:.1f} (prev={self.previous_hp:.1f})")
            print(f"   Action: {action_taken}, Rune detected: {rune_detected}")
        
        # XP gain reward
        xp_gain = current_xp - self.previous_xp
        if xp_gain > 0:
            xp_reward = xp_gain * 0.1
            reward += xp_reward
            if verbose:
                print(f"   ✓ XP gain: +{xp_reward:.4f} (from {xp_gain:.1f} xp)")
        
        # HP loss penalty
        hp_loss = self.previous_hp - current_hp
        if hp_loss > 0:
            hp_penalty = hp_loss * 0.05
            reward -= hp_penalty
            if verbose:
                print(f"   ✗ HP loss: -{hp_penalty:.4f} (lost {hp_loss:.1f} hp)")
        
        # Rune interaction bonus
        if action_taken == 5 and rune_detected:
            reward += 10.0
            if verbose:
                print(f"   🌟 RUNE HIT: +10.0 (correct interaction)")
        elif action_taken == 5 and not rune_detected:
            reward -= 0.5
            if verbose:
                print(f"   ⚠️  Wasted interact: -0.5")
        
        # Update state
        self.previous_xp = current_xp
        self.previous_hp = current_hp
        
        if verbose:
            print(f"   📊 Final reward: {reward:.4f}")
        
        return reward