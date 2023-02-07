import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
E = 0.1                 # value added to deltas
DELTA_POW = 0.5         # power factor for deltas

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def compute_delta(self, state, action, reward, next_state, done):
        # Computes the difference between the predicted and expected Q-values.
        # (Used for Prioritized Experience Replay.)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = torch.tensor([[action]]).long().to(device)       
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        reward = torch.tensor([[reward]]).float().to(device)
        done = torch.tensor([[done]]).float().to(device)
        best_local_action = torch.argmax(self.qnetwork_local(next_state).detach(), 1).unsqueeze(1)
        Q_target_next = self.qnetwork_target(next_state).gather(1, best_local_action)
        Q_target = reward + (GAMMA * Q_target_next * (1 - done))
        Q_expected = self.qnetwork_local(state).gather(1, action)
        return np.asscalar(Q_target.detach().numpy() - Q_expected.detach().numpy())     
    
    def step(self, state, action, reward, next_state, done, beta):
        # Get delta for Prioritized Experience Replay
        delta = self.compute_delta(state, action, reward, next_state, done)
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, delta)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, beta)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, beta):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, w) tuples 
            gamma (float): discount factor
            beta (float): power factor used for importance weights
        """
        states, actions, rewards, next_states, dones, importance_weights = experiences

        # Get max predicted Q values (for next states) from target model
        best_local_actions = torch.argmax(self.qnetwork_local(next_states).detach(), 1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, best_local_actions)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss((importance_weights ** (beta / 2)) * Q_expected,
                          (importance_weights ** (beta / 2)) * Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, e=E, delta_pow=DELTA_POW):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            e (float): value added to delta
            delta_pow (float): power factor for delta
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.weights = deque(maxlen=buffer_size)
        self.e = e
        self.delta_pow = delta_pow
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, delta):
        """Add a new experience to memory."""
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
        self.weights.append((np.abs(delta) + self.e) ** self.delta_pow)
        
    def compute_importance_weight(self, p):
        """Computes the importance weight from sampling probability."""
        return 1 / (p * self.batch_size)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        probs = self.weights / np.sum(self.weights)
        sampled_idxs = np.random.choice(range(len(self.memory)), size=self.batch_size,
                                        replace=False, p=probs)
        experiences = [self.memory[i] for i in sampled_idxs]

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack(
                [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        importance_weights = torch.from_numpy(
            np.vstack(
                [self.compute_importance_weight(probs[i]) for i in sampled_idxs])).float().to(device)
  
        return (states, actions, rewards, next_states, dones, importance_weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    