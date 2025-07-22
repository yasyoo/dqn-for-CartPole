# DQN for CartPole - Reinforcement Learning Project

## Overview

This project implements Deep Q-Network (DQN) algorithms to solve the CartPole-v1 environment from Gymnasium. The goal is to balance a pole on a moving cart by applying forces to the cart. The implementation includes three variants of DQN:

1. **Basic DQN**: a neural network approximation of the Q-function with ε-greedy exploration.
2. **DQN with Experience Replay**: uses a replay buffer to store and sample past experiences for more stable training.
3. **DQN with Prioritized Experience Replay**: enhances the replay buffer by prioritizing experiences based on their TD-error, improving learning efficiency.

## Features

- **Neural Network Architecture**: fully connected layers with ReLU activation for Q-value approximation.
- **ε-Greedy Exploration**: balances exploration and exploitation with a linearly decaying ε.
- **Experience Replay**: stores transitions in a buffer and samples mini-batches for training.
- **Prioritized Experience Replay**: samples transitions based on their TD-error to focus on more informative experiences.
- **Training and Evaluation**: tracks performance and stops when the agent achieves the target reward.

## Requirements

- Python 3.11
- Libraries:
  - `gymnasium`
  - `numpy`
  - `torch`
  - `matplotlib`
  - `pygame` (for rendering, optional)

Install dependencies using:
```bash
pip install gymnasium[classic-control] torch numpy matplotlib
```

## Usage

1. **Basic DQN**:
   ```python
   run_dqn(eval_schedule=250)
   ```

2. **DQN with Experience Replay**:
   ```python
   run_dqn_rb(eval_schedule=250)
   ```

3. **DQN with Prioritized Experience Replay**:
   ```python
   run_dqn_prioritized_rb(eval_schedule=250)
   ```

## Key Functions

- `create_network`: builds the neural network for Q-value approximation.
- `select_action_eps_greedy`: selects actions using ε-greedy policy.
- `compute_td_target` and `compute_td_loss`: computes TD targets and loss for training.
- `eval_dqn`: evaluates the agent's performance on a single episode.
- `sample_batch` and `sample_prioritized_batch`: samples mini-batches from the replay buffer.

## Results

The agent's performance is evaluated periodically, and training stops once the average reward over recent episodes meets or exceeds the target (200 for CartPole-v1). The prioritized replay buffer variant typically converges faster due to its focus on high-error transitions.

## Acknowledgments

This project is based on educational materials from the Applied Data Analysis course at the Faculty of Computer Science, HSE University. 

## License

This project is open-source and available under the MIT License.
