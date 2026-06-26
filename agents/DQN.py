import torch
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F
from typing import List, Tuple, Dict, Optional
import random

from tqdm import tqdm

from utils.device import device, autocast, grad_scaler
from env.state import TensorState
from env.environment import ParallelEnvironment
from agents.basic_agent import BaseAgent, SummaryWriter
from modules.networks import ActionNet


class DQNAgent(BaseAgent):
    def __init__(self, batch_size: int = 256, horizon: int = 400, num_training_steps: int = 100, num_samples: int = 512, buffer_capacity: int = 4, learning_rate: float = 1e-4, gamma: float = 0.9, epsilon: float = 0.8, target_update: int = 64):
        super().__init__(buffer_capacity, num_samples, batch_size, horizon, 'item')
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.update_counter = 0
        self.num_training_steps = num_training_steps

        self.active_net = ActionNet().to(device)
        self.stable_net = ActionNet().to(device)
        self.stable_net.load_state_dict(self.active_net.state_dict())
        self.stable_net.eval()

        self.optimizer = torch.optim.AdamW(self.active_net.parameters(), lr=learning_rate, weight_decay=0.001)
    
    def eval(self): self.active_net.eval()
    def train(self): self.active_net.train()

    def update_stable_net(self):
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.stable_net.load_state_dict(self.active_net.state_dict())
            self.stable_net.eval()

    def select_actions(self, state: TensorState, evaluate: bool) -> LongTensor:
        B = len(state.items)
        if evaluate or random.random() < self.epsilon:
            with torch.no_grad():
                with autocast:
                    q_values: Tensor = self.active_net(state)
            return torch.argmax(q_values, dim=1)
        else:
            return torch.randint(0, self.num_joint_actions, (B,), dtype=torch.int, device=device)

    def train_step(self, writer: SummaryWriter, global_step: int) -> Tuple[float, int]:
        if len(self.buffer) < self.num_samples: return 0.0
        total_loss = 0.0

        for i in tqdm(range(1, 1 + self.num_training_steps), desc='Training', dynamic_ncols=True, leave=False):
            states, actions, rewards, dones, next_states = self.buffer.sample(self.num_samples)

            self.optimizer.zero_grad()

            with autocast:
                with torch.no_grad():
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device).float() * self.active_net.state_encoder.value_scale
                    dones = dones.to(device)
                    next_states = next_states.to(device)

                    next_q_values: Tensor = self.stable_net(next_states)
                    next_max_q = next_q_values.max(dim=1).values
                    target_q_values = rewards + self.gamma * next_max_q * torch.logical_not(dones).float()

                q_values: Tensor = self.active_net(states)
                action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(action_q_values, target_q_values)
            
            grad_scaler.scale(loss).backward()
            grad_scaler.step(self.optimizer)
            grad_scaler.update()

            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.active_net.parameters(), max_norm=1.0)
            # self.optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item
            self.update_stable_net()

            global_step += 1
            writer.add_scalar('training_loss', loss, global_step)

        avg_loss = total_loss / self.num_training_steps
        return avg_loss, global_step

    def get_state_dict(self):
        state_dict = {
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'target_update': self.target_update,
            'update_counter': self.update_counter,
            'active_net': self.active_net.state_dict(),
            'stable_net': self.stable_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return state_dict

    def load_from_state_dict(self, state_dict):
        self.gamma: float = state_dict['gamma']
        self.epsilon: float = state_dict['epsilon']
        self.target_update: int = state_dict['target_update']
        self.update_counter: int = state_dict['update_counter']
        self.active_net.load_state_dict(state_dict['active_net'])
        self.stable_net.load_state_dict(state_dict['stable_net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])


