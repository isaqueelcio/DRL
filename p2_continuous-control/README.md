
# 🤖 Continuous Control – DDPG Agent

This repository implements a Deep Deterministic Policy Gradient (DDPG) agent to solve the Unity Reacher environment using PyTorch.

Training uses the **DDPG algorithm** implemented in `ddpg_agent.py` and the models in `model.py`.

---

## 🧪 Algorithm Description

This project uses the **DDPG (Deep Deterministic Policy Gradient)** algorithm, an actor-critic method suitable for continuous action spaces.

### Model Architectures


#### 🎬 Actor Network
- **Input:** 33-dimensional state
- **Hidden Layers:**
  - Linear(33 → 256) → LayerNorm → ReLU
  - Linear(256 → 256) → LayerNorm → ReLU
- **Output:** Linear(256 → 4) → Tanh (bounded continuous actions)

#### 🧮 Critic Network
- **Input:** 33-dimensional state + 4-dimensional action
- **Hidden Layers:**
  - Linear(33 → 256) → LayerNorm → ReLU
  - 🔗 **Then concatenates action (4-dim), resulting in 256 + 4 = 260 features**
  - Linear(260 → 256) → LayerNorm → ReLU
- **Output:** Linear(256 → 1) → Q-value estimate

### Hyperparameters

| Parameter         | Value     |
|------------------|-----------|
| Replay Buffer     | 1e6       |
| Batch Size        | 128       |
| Gamma (γ)         | 0.99      |
| Tau (τ)           | 1e-3      |
| Actor LR          | 1e-4      |
| Critic LR         | 1e-3      |
| Weight Decay      | 0         |
| Update Every      | 20 steps  |
| Updates per Step  | 5         |
| Noise Sigma       | 0.2       |

---

## 📁 Files

- `Continuous_Control.ipynb`: main training notebook
- `ddpg_agent.py`: DDPG agent class
- `model.py`: Actor & Critic models
- `checkpoint_actor.pth`: trained actor weights
- `checkpoint_critic.pth`: trained critic weights

---
