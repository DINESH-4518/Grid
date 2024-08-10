GridWorld DQN Agent
This project implements a Deep Q-Network (DQN) agent to solve a custom GridWorld environment. The agent is trained to navigate through a grid filled with obstacles, reaching the goal while avoiding traps.

Features
GridWorld Environment: A 5x5 grid where the agent starts at the top-left corner and must reach the bottom-right corner (goal). The grid contains obstacles, making the path to the goal challenging.

Customizable Environment: The grid and obstacle placements can be modified to create different scenarios and challenges for the agent.

Deep Q-Network: The DQN agent is implemented using PyTorch, leveraging a neural network to approximate Q-values and make decisions.

Training and Testing: The agent is trained over multiple episodes, learning the optimal path to the goal. After training, the agent can be tested in the environment to evaluate its performance.
