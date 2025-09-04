Dueling DQN: Deep Reinforcement Learning Agent

This repository implements a Dueling Deep Q-Network (Dueling DQN) agent for reinforcement learning tasks in a Unity environment. The Dueling DQN is an improved variant of the standard Deep Q-Network (DQN) that separates the estimation of state-value and advantage for each action, which stabilizes learning and improves performance.




Table of Contents

Project Overview

Installation

Usage

Code Structure

Model Architecture

Training Process

Results

References

License



Project Overview

The Dueling DQN agent interacts with a Unity-based environment to learn optimal actions for navigation and task completion. It uses reinforcement learning principles, including:

Q-learning: Update Q-values using the Bellman equation.

Experience Replay: Store past experiences (state, action, reward, next_state, done) and train in batches.

Epsilon-Greedy Policy: Balance exploration and exploitation.



Installation

Clone the repository and install dependencies:

git clone https://github.com/Onkar404/Dueling_dqn.git
cd Dueling_dqn
pip install -r requirements.txt


Make sure your Unity environment is set up and compatible with the Python API.



Usage

To train the agent:

python Navigation.py


This script initializes the Unity environment and starts training the Dueling DQN agent.

Model checkpoints are saved automatically during training (e.g., checkpoint.pth).

To test a trained model:

python Navigation.py --load_checkpoint checkpoint.pth




Code Structure

Navigation.py: Main script to run training and testing.

dqn_agent_banana.py: Dueling DQN agent class implementing:

step(): Update agent per environment step.

act(): Choose action using epsilon-greedy policy.

learn(): Update Q-network using experiences.

model.py: Defines the Dueling DQN neural network:

Input: state vector from environment

Hidden layers: fully connected layers

Output: Q-values for each action, computed as V(s) + (A(s,a) - mean(A(s,·)))

checkpoints: Saved model weights during training.

.ipynb_checkpoints/: Optional Jupyter notebook experiments.





Model Architecture

Dueling DQN separates value and advantage streams:

Input (state) --> FC layers --> 
    |--> Value Stream (V(s)) --> FC --> Output
    |--> Advantage Stream (A(s,a)) --> FC --> Output
Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))


Advantage Stream: Measures the benefit of taking a specific action.

Value Stream: Measures how good it is to be in a particular state.

This helps the agent learn which states are valuable regardless of the action.

Training Process

Initialize the environment and agent.

For each episode:

Select action using epsilon-greedy policy.

Observe reward and next state.

Store (state, action, reward, next_state, done) in replay buffer.

Sample random minibatch from buffer and update Q-network.

Gradually decay epsilon to reduce exploration over time.

Save model checkpoints periodically.

Loss Function:

loss = F.mse_loss(Q_expected, Q_target)


Q_expected: Q-values from current network.

Q_target: Target Q-values using Bellman equation.

Results

The Dueling DQN agent learns to maximize cumulative reward in the environment.

Training performance improves over episodes, showing stable convergence due to the dueling architecture.



References

Wang, Z., et al. Dueling Network Architectures for Deep Reinforcement Learning. https://arxiv.org/abs/1511.06581

Unity ML-Agents Toolkit: https://github.com/Unity-Technologies/ml-agents

License

This project is licensed under the MIT License. See the LICENSE file for details.
