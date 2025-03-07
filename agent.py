import torch
import random
import numpy as np
from collections import deque
from neural import AgentNet, MODEL_FLAG_ONLINE, MODEL_FLAG_TARGET

class Agent:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None, epsilon=1.0):
        my_rig_factor = 0.9
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=int(100_000 * my_rig_factor))
        self.batch_size = 32

        self.exploration_rate = epsilon
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.02
        self.gamma = 0.99

        self.curr_step = 0
        self.burnin = int(100_000 * my_rig_factor)  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1_000  # no. of experiences between Q_target & Q_online sync (tau)

        self.save_every = 200_000   # no. of experiences between saving Agent Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        self.use_mps_device = torch.backends.mps.is_available()
        if self.use_cuda:
            self.device = 'cuda'
        elif self.use_mps_device:
            self.device = 'mps:0'
        else:
            self.device = 'cpu'
        torch.device(self.device)
        torch.set_default_device(self.device)
        print(f"Using device: {torch.get_default_device()}")

        # Script the network and use it directly
        self.net = torch.jit.script(AgentNet((4, 84, 84), action_dim))
        self.net = self.net.to(device=self.device)
        if checkpoint:
            print(f"Loading: {checkpoint}")
            self.load(checkpoint)

        self.learning_rate = 0.00025
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state: A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Agent will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            state = state.unsqueeze(0)
            action_values = self.net(state, MODEL_FLAG_ONLINE)  # Use 0 for 'online'
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state,
        next_state,
        action (int),
        reward (float),
        done (bool)
        """

        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor([action], dtype=torch.long, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float, device=self.device)
        done = torch.tensor([done], dtype=torch.bool, device=self.device)

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, MODEL_FLAG_ONLINE)[np.arange(0, self.batch_size), action]  # Use 0 for 'online'
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q_online = self.net(next_state, MODEL_FLAG_ONLINE)  # Use 0 for 'online'
        best_action = torch.argmax(next_state_Q_online, axis=1)
        next_Q = self.net(next_state, MODEL_FLAG_TARGET)[np.arange(0, self.batch_size), best_action]  # Use 1 for 'target'
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.update_target_network()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def save(self):
        save_path = self.save_dir / f"agent_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"Agent Net saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=(self.device))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
        print(f"Loaded model from {load_path} with exploration rate {self.exploration_rate}")