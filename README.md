# Deep Reinforcement Learning Nanodegree - Udacity

This repository contains projects completed as part of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## 🎯 Projects

### Project 1: Navigation
**Goal:** Train an agent to navigate a virtual environment and collect yellow bananas while avoiding blue ones.

- **Algorithm:** Deep Q-Network (DQN)
- **State Space:** 37 dimensions
- **Action Space:** 4 discrete actions
- **Success Criteria:** Average score of +13 over 100 episodes
- **Location:** `p1_navigation/`

[View Project Details](p1_navigation/README.md)

---

### Project 2: Continuous Control
**Goal:** Train a robotic arm to maintain its position at target locations.

- **Algorithm:** Deep Deterministic Policy Gradient (DDPG)
- **State Space:** 33 dimensions (continuous)
- **Action Space:** 4 dimensions (continuous)
- **Success Criteria:** Average score of +30 over 100 episodes
- **Location:** `p2_continuous-control/`

[View Project Details](p2_continuous-control/README.md)

---

### Project 3: Collaboration and Competition
**Goal:** Train two agents to play tennis collaboratively, keeping the ball in play.

- **Algorithm:** Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
- **State Space:** 24 dimensions per agent
- **Action Space:** 2 dimensions per agent (continuous)
- **Success Criteria:** Average score of +0.5 over 100 episodes
- **Status:** ✅ Solved at Episode 1077 (average score: 0.509)
- **Location:** `p3_collab-compet/`

[View Project Details](p3_collab-compet/README.md)

---

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook
- Unity ML-Agents

### Installation

1. Clone this repository:
```bash
git clone https://github.com/isaqueelcio/DRL.git
cd DRL
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv drlnd
# On Windows:
drlnd\Scripts\activate
# On Linux/Mac:
source drlnd/bin/activate
```

3. Install dependencies for each project:
```bash
cd p1_navigation
pip install -r requirements.txt
```

Repeat for other projects as needed.

---

## 📚 Course Content

This nanodegree covers:

- **Value-Based Methods:** DQN, Double DQN, Dueling DQN, Prioritized Experience Replay
- **Policy-Based Methods:** REINFORCE, PPO, GAE, Actor-Critic (DDPG)
- **Multi-Agent RL:** MADDPG, Markov Games, AlphaZero

---

## 🎓 Certificate

This repository represents the coursework completed for the Udacity Deep Reinforcement Learning Nanodegree.

---

## 📝 License

This project is licensed under the MIT License - see individual project directories for details.

---

## 🙏 Acknowledgments

- Udacity for providing the course materials and project templates
- Unity Technologies for the ML-Agents Toolkit
- The research papers and authors whose work these implementations are based on