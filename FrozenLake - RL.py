

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


# Creating a virtual environment for developing a game

# In[3]:


class FrozenLake:                               # creating FrozenLake environment
    def __init__(self, grid_size=(5, 5)):
        self.grid_size = grid_size
        self.grid = np.zeros(self.grid_size, dtype=np.int32)
        self.agent_position = (0, 0)
        self.goal_position = (4, 4)

        # Setting holes in the environment
        self.grid[1, 0] = -1
        self.grid[4, 0] = -1
        #self.grid[0, 1] = -1
        self.grid[1, 1] = -1
        self.grid[3, 4] = -1

    def reset(self):                            # restarting the game
        self.agent_position = (0, 0)
        return self.agent_position

    def step(self, action):
        done = False
        reward = 0

        # Updating the position
        if action == 0:  # Move up
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1:  # Move down
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 2:  # Move left
            new_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 3:  # Move right
            new_position = (self.agent_position[0], self.agent_position[1] + 1)
        elif action == 4:  # Move top right
            new_position = (self.agent_position[0]+1, self.agent_position[1] + 1)
        elif action == 5:  # Move bottom right
            new_position = (self.agent_position[0]-1, self.agent_position[1] + 1)
        elif action == 6:  # Move top left
            new_position = (self.agent_position[0]+1, self.agent_position[1] - 1)
        elif action == 7:  # Move bottom left
            new_position = (self.agent_position[0]-1, self.agent_position[1] - 1)
        else:
            raise ValueError("Invalid action.")

        # Checking if the new position is within the grid boundaries
        if (
            0 <= new_position[0] < self.grid_size[0]
            and 0 <= new_position[1] < self.grid_size[1]
        ):
            self.agent_position = new_position

        # Checking if the agent reached the goal or hit an obstacle
        if self.agent_position == self.goal_position:
            reward = 10
            done = True
        elif self.grid[self.agent_position] == -1:
            reward = -5
            done = True
        else:
            reward = -1

        return self.agent_position, reward, done

                
    def render(self):
        # Viewing the current Lake with the agent's position and goal
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if (i, j) == self.agent_position:
                    print("A", end=" ")
                elif (i, j) == self.goal_position:
                    print("G", end=" ")
                elif self.grid[i, j] == -1:
                    print("X", end=" ")
                else:
                    print("-", end=" ")
            print()
        print()


# Creating a neural network archichecture for implementing DQN Algorithm

# In[4]:


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Creating an Agent for playing the game

# In[5]:


class DQN:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, target_update_freq = 10,  buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        
        # Intialising the buffer memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Intialising the neural networks
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:                            # Exploration
            return np.random.choice(self.action_size)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)    # Exploitation
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
            
    def train(self, env, num_episodes=1000):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                total_reward += reward
                
                self.store_transition(state, action, reward, next_state, done)
                
                if len(self.replay_buffer) >= self.batch_size:
                    minibatch = random.sample(self.replay_buffer, self.batch_size)

                    states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*minibatch)
                
                    state_tensor = torch.tensor(states_batch, dtype=torch.float32)
                    action_tensor = torch.tensor(actions_batch, dtype=torch.int64)
                    reward_tensor = torch.tensor(rewards_batch, dtype=torch.float32)
                    next_state_tensor = torch.tensor(next_states_batch, dtype=torch.float32)
                    done_tensor = torch.tensor(dones_batch, dtype=torch.float32)
                    
                    q_values = self.q_network(state_tensor)
                    next_q_values = self.target_network(next_state_tensor)

                    targets = reward_tensor + (1.0 - done_tensor) * self.gamma * torch.max(next_q_values, dim=1)[0]

                    q_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                    loss = F.smooth_l1_loss(q_values, targets)
                    
                    # Back Propagation and weights updation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                state = next_state
                
            if episode % self.target_update_freq == 0:
                self.update_target_network()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            print("Episode {}: Total Reward = {}".format(episode, total_reward))


# Intialising Environment and Agent to play the game

# In[6]:


env = FrozenLake()

# Create the DQN agent
state_size = 2
action_size = 8
dqn_agent = DQN(state_size, action_size)

# Train the DQN agent
num_episodes = 1000
dqn_agent.train(env, num_episodes)

# Test the trained agent
state = env.reset()
done = False

while not done:
    env.render()
    action = dqn_agent.choose_action(state)
    next_state, reward, done = env.step(action)
    state = next_state
    #print("Total Reward = {}".format(reward))
    
env.render()
print("Total Reward = {}".format(reward))


# In[ ]:




