# Project: Navigation

## 🏑 Udacity Deep Reinforcement Learning Nanodegree

## 📌 Objective

The goal of this project is to train an agent to navigate a virtual environment and collect yellow bananas while avoiding blue ones.

- 🟡 **+1 reward** for collecting a yellow banana  
- 🔵 **-1 reward** for collecting a blue banana

✅ The environment is **considered solved** when the agent achieves an **average score of at least +13**.
---

## 🎮 Environment Details

The environment is built using [Unity ML-Agents Toolkit (v0.4)](https://github.com/Unity-Technologies/ml-agents).  
The agent observes a **state vector of length 37**, which includes:


The **action space** is discrete with **4 possible actions**:

| Action | Description   |
|--------|---------------|
| 0      | Move forward  |
| 1      | Move backward |
| 2      | Turn left     |
| 3      | Turn right    |

Only **one agent** is trained in this environment.

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/isaqueelcio/DRL.git
cd DRL/p1_navigation/
```

### 2. Install dependencies

Make sure you have Python 3.10+ and install the requirements:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:
```
torch>=1.8.1
numpy
matplotlib
unityagents==0.4.0
jupyter
```

> Important: This project uses `unityagents==0.4.0`, compatible with the Unity environment version provided by Udacity.

### 3. Run the notebook

```bash
jupyter notebook
```

Then open and run all cells in:

```
Navigation.ipynb
```

Training will begin and progress will be printed to the screen.

---

## 🧠 Learning Algorithm

We use **Deep Q-Learning (DQN)** to train the agent.

### Architecture

- Input: 37 (state size)
- Hidden layers: 64 → 64
- Output: 4 (action size)
- Activations: ReLU

### Techniques

- Epsilon-Greedy Action Selection  
- Experience Replay Buffer  
- Fixed Q-Targets  
- Soft Update of Target Network

---

## 📁 Repository Structure

```
📆 projeto-navigation/
├── model.py             # Neural network definition
├── dqn_agent.py         # Agent and replay buffer
├── Navigation.ipynb     # Main notebook (training)
├── checkpoint.pth       # Trained weights
├── Report_navigation.md # Explanation of approach/results
├── requirements.txt     # Project dependencies
└── README.md            # Project overview and setup
```


