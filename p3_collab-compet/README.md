# 🎾 Collaboration and Competition – Multi-Agent DDPG

This repository implements a **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** to solve the Unity Tennis environment using PyTorch.

Training uses the **MADDPG algorithm** implemented in `maddpg_agent.py` with neural network models defined in `model.py`.

---

## 🏑 Udacity Deep Reinforcement Learning Nanodegree

## 📌 Objective

The goal of this project is to train two agents to play tennis collaboratively in a virtual environment. The agents must keep the ball in play for as many timesteps as possible.

- ✅ **+0.1 reward** when an agent hits the ball over the net  
- ❌ **-0.01 reward** when an agent lets the ball hit the ground or hits it out of bounds

🎯 The environment is **considered solved** when the agents achieve an **average score of at least +0.5 over 100 consecutive episodes** (taking the maximum score between both agents per episode).

---

## 🎮 Environment Details

The environment is built using [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).

### State Space
The **state space** is **24-dimensional** for each agent, containing information about:
- Position and velocity of the ball
- Position and velocity of the racket

### Action Space
The **action space** is **2-dimensional** and continuous, with values between -1 and 1:
- Movement toward or away from the net
- Jumping

### Agents
**Two agents** are trained simultaneously in this collaborative/competitive environment.

### Solving Criteria
The task is episodic. After each episode, the rewards for each agent are summed (without discounting) to get a score for each agent. The **maximum score** between both agents is then taken as the episode score.

The environment is considered solved when the average score over 100 consecutive episodes is at least **+0.5**.

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/isaqueelcio/DRL.git
cd DRL/p3_collab-compet/
```

### 2. Install dependencies

Make sure you have Python 3.10+ and install the required packages:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```
torch>=1.8.1
numpy
matplotlib
jupyter
unityagents==0.4.0
```

### 3. Download the Unity Environment

Download the Tennis environment for your operating system:

- **Windows (64-bit)**: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
- **Linux**: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- **macOS**: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)

Extract the downloaded file to the `p3_collab-compet/` directory.

### 4. Run the training

```bash
jupyter notebook
```

Then open and execute all cells in `Tennis_03.ipynb`.

### 5. Trained Model

The pre-trained MADDPG model that solved the environment is available in:
- `checkpoint_maddpg_actor_0.pth` (Agent 1 Actor)
- `checkpoint_maddpg_actor_1.pth` (Agent 2 Actor)
- `checkpoint_maddpg_critic_0.pth` (Agent 1 Critic)
- `checkpoint_maddpg_critic_1.pth` (Agent 2 Critic)

The environment was **solved at Episode 1077 (average score: 0.509)** over 100 consecutive episodes.

---

## 📁 Files Included

| File                            | Description                                    |
|---------------------------------|------------------------------------------------|
| `Tennis_03.ipynb`               | **MADDPG training notebook**                   |
| `maddpg_agent.py`               | **MADDPG agent implementation**                |
| `model.py`                      | Actor and Critic network architectures         |
| `checkpoint_maddpg_actor_0.pth` | **Saved MADDPG Actor weights (Agent 0)**       |
| `checkpoint_maddpg_actor_1.pth` | **Saved MADDPG Actor weights (Agent 1)**       |
| `checkpoint_maddpg_critic_0.pth`| **Saved MADDPG Critic weights (Agent 0)**      |
| `checkpoint_maddpg_critic_1.pth`| **Saved MADDPG Critic weights (Agent 1)**      |
| `rewards_plot_maddpg.png`       | **Training progress visualization**            |
| `requirements.txt`              | Python dependencies                            |
| `README.md`                     | Setup and execution instructions (this file)   |
| `Report.md`                     | Detailed project report                        |

---

## 🧠 Algorithm

This project uses **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**, an extension of DDPG designed specifically for multi-agent environments with continuous action spaces.

### Key Features:

- **Centralized Training, Decentralized Execution**: Each agent has its own Actor network but learns from a centralized Critic that observes all agents
- **Independent Actor Networks**: Each of the 2 agents has its own policy network for decentralized execution
- **Centralized Critics**: Each agent's Critic has access to all agents' states and actions during training
- **Shared Replay Buffer**: Experiences from both agents are stored in a common replay buffer
- **Ornstein-Uhlenbeck Noise with Decay**: Exploration noise that decreases over time (decay rate: 0.9995)
- **Target Networks**: Soft updates (τ=1e-3) for stability
- **Gradient Clipping**: Prevents exploding gradients

For detailed information about the algorithm, hyperparameters, network architectures, and training results, see [Report.md](Report.md).

---

## 📊 Results

The agents successfully solved the environment, achieving an average score of +0.5 over 100 consecutive episodes. For detailed training metrics and analysis, please refer to [Report.md](Report.md).

---

## 📄 License

This project is part of the Udacity Deep Reinforcement Learning Nanodegree program.
