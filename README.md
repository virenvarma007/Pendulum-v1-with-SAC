# 🧠 Solving Pendulum-v1 with Soft Actor-Critic (SAC)

A reinforcement learning project that implements the **Soft Actor-Critic (SAC)** algorithm to solve the classic **Pendulum-v1** continuous control problem from OpenAI Gym.

## 📌 Problem Statement

The **Pendulum-v1** environment involves applying torque to a frictionless pendulum to keep it upright.

- **State Space:** `[cos(θ), sin(θ), θ̇]`  
- **Action Space:** Continuous scalar torque in `[-2, 2]`
- **Reward Function:**  
  \[
  r = -(\theta^2 + 0.1 \cdot \dot{\theta}^2 + 0.001 \cdot \tau^2)
  \]

The agent must **stabilize the pendulum**, **minimize movement**, and **reduce energy use**.

## 🚀 Why Soft Actor-Critic?

Traditional algorithms like DDPG and PPO often struggle with:

- Overestimation of Q-values
- Premature convergence
- Lack of exploration in continuous action spaces

**SAC** overcomes these via:

- 🧑‍🎓 **Stochastic Actor:** Outputs mean and standard deviation of action
- 🧮 **Dual Critics (Q1, Q2):** Prevents overestimation bias
- 🔁 **Replay Buffer:** Supports stable and generalizable learning
- 🔥 **Entropy Regularization:** Encourages effective exploration

## ⚙️ Core Components

- **Actor Network:** Learns stochastic Gaussian policy \(\pi(a|s)\)
- **Critic Networks (Q1, Q2):** Learn action-value functions
- **Temperature Coefficient (α):** Balances exploration vs. exploitation
- **Replay Buffer:** Stores transitions for training

## 🔄 Training Loop

1. Collect transition: \((s, a, r, s')\)
2. Store in replay buffer
3. Sample mini-batch of transitions
4. Update critics using Bellman error
5. Update actor to maximize expected reward + entropy
6. Adjust entropy coefficient α (optional)

## 📈 Results

- Early training shows low rewards (~−1800) due to high-entropy policy
- Rewards improve gradually after episode 40
- Stabilize near −120 by episode 100
- Demonstrates smooth, efficient control of the pendulum

## 📁 File Structure

├── code.ipynb # Jupyter notebook implementation of SAC for Pendulum-v1 
├── README.md # You’re here!


## 🛠️ Dependencies

- Python ≥ 3.7
- PyTorch ≥ 1.9
- gym ≥ 0.26
- NumPy
- Matplotlib

Install via:
```bash
pip install torch gym numpy matplotlib

