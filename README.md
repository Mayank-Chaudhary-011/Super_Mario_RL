# Super Mario AI using Double Deep Q-Network (DDQN)

This project implements an AI agent that learns to play **Super Mario Bros** using **Deep Reinforcement Learning (DDQN)**.

---

## Key Concepts

- Reinforcement Learning (RL)
- Deep Q-Network (DQN)
- **Double DQN (DDQN)** ✅
- Experience Replay Buffer
- Target Network
- Epsilon-Greedy Strategy
- Convolutional Neural Networks (CNNs)

---

## Environment

- Game: `SuperMarioBros-1-1`
- Library: `gym-super-mario-bros`
- Action Space: Reduced (RIGHT_ONLY)
- Observations: Processed frames (grayscale, resized, stacked)

---

## 📁 Project Structure

```
Mario/
│
├── agent.py              # DDQN agent logic
├── model.py              # CNN-based Q-network
├── replay_buffer.py      # Experience replay memory
├── environment.py        # Environment wrappers
├── train.py              # Training loop
├── utils.py              # Logging / helper functions
├── wrappers.py           # Preprocessing (frame skip, resize, stack)
├── parameters.yaml       # Hyperparameters
└── README.md
```

---

## How It Works

1. Agent observes stacked game frames
2. Chooses action using epsilon-greedy policy
3. Stores experience `(state, action, reward, next_state, done)`
4. Samples mini-batch from replay buffer
5. Uses **DDQN** to reduce overestimation:
   - Online network → selects action
   - Target network → evaluates action

6. Updates Q-values using backpropagation

---

## Training Details

- Algorithm: **Double Deep Q-Network (DDQN)**
- Framework: PyTorch
- Optimizer: Adam
- Loss: Huber Loss (Smooth L1)
- Discount Factor (γ): 0.99
- Replay Buffer Size: Large (to stabilize learning)
- Target Network Update: Periodic

---

## ▶ Installation

⚠️ Recommended: Python 3.9 (for compatibility)

```bash
pip install torch torchvision
pip install gym==0.26.2
pip install gym-super-mario-bros
pip install nes-py
pip install opencv-python numpy matplotlib
pip install pyyaml
```

---

## ▶ Run Training

```bash
python train.py --train
```

---

## Common Issues (Important)

### Gym API Error (4 vs 5 values)

Fix inside environment wrapper:

```python
obs, reward, done, info = env.step(action)
terminated = done
truncated = False
```

---

### Gym + NumPy warnings

You can ignore OR downgrade NumPy:

```bash
pip install numpy==1.23.5
```

---

## Future Improvements

- Prioritized Experience Replay
- Dueling DQN
- Multi-step learning
- Better reward shaping
- Save & load trained models

---

## Results

- Agent learns to:
  - Move right consistently
  - Avoid obstacles
  - Survive longer over time

(Add graphs / GIFs here later for better presentation)

---

## Note

This project demonstrates a full implementation of **Deep Reinforcement Learning applied to a real game environment**, focusing on stability improvements using **DDQN**.
