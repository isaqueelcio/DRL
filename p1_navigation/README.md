# Project: Navigation

### Udacity Deep Reinforcement Learning Nanodegree

## 📌 Objective

The goal of this project is to train an agent to navigate in a virtual environment and collect yellow bananas while avoiding blue ones. The agent receives:

- **+1 reward** for collecting a yellow banana  
- **-1 reward** for collecting a blue banana

The task is **solved** when the agent achieves an **average score of at least +13** over **100 consecutive episodes**.

---

## 🎮 Environment Details

The environment is provided by [Unity ML-Agents Toolkit (v0.4)](https://github.com/Unity-Technologies/ml-agents), and the agent observes a **state vector of length 37** that includes:

- The agent’s velocity
- A ray-based perception of nearby objects

**Action space** is discrete with 4 possible actions:

| Action | Description      |
|--------|------------------|
|   0    | Move forward     |
|   1    | Move backward    |
|   2    | Turn left        |
|   3    | Turn right       |

Only **one agent** is trained.

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/seu-usuario/projeto-navigation.git
cd projeto-navigation
