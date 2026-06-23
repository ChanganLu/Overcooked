import torch
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F
import random

from tqdm import tqdm

from utils.device import device, autocast, grad_scaler
from env.state import TensorState
from env.environment import ParallelEnvironment
from agents.basic_agent import BaseAgent
from modules.networks import ActionNet


class DQNAgent(BaseAgent):
    def __init__(self, learning_rate: float = 1e-3, gamma: float = 0.9, epsilon: float = 0.8, target_update: int = 64, buffer_capacity: int = 256, num_samples: int = 8, batch_size: int = 256, horizon: int = 400):
        super().__init__(buffer_capacity, num_samples, batch_size, horizon)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.update_counter = 0

        self.active_net = ActionNet().to(device)
        self.stable_net = ActionNet().to(device)
        self.stable_net.load_state_dict(self.active_net.state_dict())
        self.stable_net.eval()

        self.optimizer = torch.optim.AdamW(self.active_net.parameters(), lr=learning_rate, weight_decay=0.001)
    
    def eval(self): self.active_net.eval()
    def train(self): self.active_net.train()

    def select_actions(self, state: TensorState, evaluate: bool) -> LongTensor:
        B = len(state.items)
        if evaluate or random.random() < self.epsilon:
            with torch.no_grad():
                with autocast:
                    q_values: Tensor = self.active_net(state)
            return torch.argmax(q_values, dim=1)
        else:
            return torch.randint(0, self.num_joint_actions, (B,), dtype=torch.long, device=device)

    def train_step(self) -> float:
        if len(self.buffer.buffer) < self.num_samples: return 0.0

        B = self.batch_size
        N = self.horizon
        chunk_size = 24
        num_chunks = (N + chunk_size - 1) // chunk_size

        self.train()
        samples = self.buffer.sample(self.num_samples)
        total_loss = 0.0
        pbar = tqdm(total = self.num_samples * num_chunks, desc='Training Network [Loss = ??.??????]', dynamic_ncols=True, leave=False)

        for all_states, all_actions, all_rewards in samples:
            for chunk_id in torch.randperm(num_chunks).tolist():
                chunk_id: int
                beg = chunk_id * chunk_size * B
                end = min((chunk_id + 1) * chunk_size, N) * B
                is_last = end == B * N
                
                states = all_states[beg:end].to(device)
                actions = all_actions[beg:end].to(device)
                rewards = all_rewards[beg:end].to(device) * self.active_net.state_encoder.value_scale
                next_states = all_states[beg+B:end].to(device) if is_last else all_states[beg+B:end+B].to(device)

                self.optimizer.zero_grad()

                with torch.no_grad():
                    next_q_values: Tensor = self.stable_net(next_states)
                    next_max_q = next_q_values.max(dim=1).values
                    if is_last:
                        target_q_values = rewards.clone()
                        target_q_values[:-B] += self.gamma * next_max_q
                    else:
                        target_q_values = rewards + self.gamma * next_max_q

                q_values: Tensor = self.active_net(states)
                action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                loss = F.mse_loss(action_q_values, target_q_values)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.active_net.parameters(), max_norm=1.0)
                self.optimizer.step()

                loss_item = loss.item()
                total_loss += loss_item * (end - beg)
                pbar.set_description(f'Training Network [Loss = {loss_item:9.6f}]')
                pbar.update()

                self.update_counter += 1
                if self.update_counter % self.target_update == 0:
                    self.stable_net.load_state_dict(self.active_net.state_dict())
                    self.stable_net.eval()
        
        pbar.close()
        avg_loss = total_loss / (self.num_samples * B * N)
        return avg_loss

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


