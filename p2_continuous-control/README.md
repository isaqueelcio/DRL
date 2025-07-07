
# 🤖 Continuous Control – DDPG Agent

This repository implements a Deep Deterministic Policy Gradient (DDPG) agent to solve the Unity Reacher environment using PyTorch.

Training uses the **DDPG algorithm** implemented in `ddpg_agent.py` and the models in `model.py`.

# 📄 Report.md – Continuous Control (Udacity DRL Nanodegree)

## 📌 1. Project Overview

🎯 **Goal:** Achieve an average score of **+30**.

✅ **Reward:** The agent receives **+0.1** for every timestep its hand remains within the target zone.

---
## 🎮 Environment Details

The environment is built with [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).\
The **state space** is 33-dimensional and the **action space** is 4-dimensional with continuous values between -1 and 1.\
Only **one agent** is used in this setup.

---

## ⚙️ Getting Started

### 2. Clone the repository

```bash
git clone https://github.com/isaqueelcio/DRL.git
cd DRL/p2_continuous_control/
```
### 3. Install dependencies

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
### 3. Run the training

```bash
jupyter notebook
```

Then open and execute all cells in:

```
Continuous_Control.ipynb
```
## 📁 4. Files Included

| File                    | Description                         |
|-------------------------|-------------------------------------|
| `Continuous_Control.ipynb` | Training script and environment logic |
| `ddpg_agent.py`         | DDPG agent implementation           |
| `model.py`              | Actor and Critic network definitions |
| `checkpoint_actor.pth`  | Saved Actor weights                 |
| `checkpoint_critic.pth` | Saved Critic weights                |
| `README.md`             | Setup and execution instructions    |
| `report.md`             | This report                         |

