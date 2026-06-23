import torch
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F

from tqdm import tqdm

from utils.device import device, autocast, grad_scaler
from env.state import TensorState
from env.environment import ParallelEnvironment
from agents.basic_agent import BaseAgent
from modules.networks import ActionNet


class DQNAgent(BaseAgent):
    def __init__(self, learning_rate: float = 1e-3, gamma: float = 0.99, epsilon: float = 0.2, target_update: int = 10, buffer_capacity: int = 1024, num_samples: int = 8):
        super().__init__(buffer_capacity, num_samples)
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
        if evaluate:
            return torch.randint(0, self.num_joint_actions, (B,), dtype=torch.long, device=device)
        else:
            with torch.no_grad():
                with autocast:
                    q_values: Tensor = self.active_net(state)
            return torch.argmax(q_values, dim=1)

    def train_step(self) -> float:
        if len(self.buffer.buffer) < self.num_samples: return 0.0

        self.train()
        samples = self.buffer.sample(self.num_samples)
        avg_loss = 0.0

        pbar = tqdm(total = self.num_samples * len(samples[0][2]), desc='Training Network [Loss = ??.??????]', dynamic_ncols=True, leave=False)

        for state_list, action_list, reward_list in samples:
            states = state_list[0].clone().to(device)
            arange = torch.arange(0, len(states.items), dtype=torch.long, device=device)
            sample_loss = 0.0
            horizon = len(action_list)
            for i in range(horizon):
                actions = action_list[i].clone().to(device)
                rewards = reward_list[i].clone().to(device).float() * self.active_net.state_encoder.value_scale
                next_states = state_list[i + 1].clone().to(device)

                self.optimizer.zero_grad()

                q_values: Tensor = self.active_net(states)
                action_q = q_values[arange, actions]

                with torch.no_grad():
                    next_q_values: Tensor = self.stable_net(next_states)
                    next_max_q = next_q_values.max(dim=1).values
                    target_q = rewards + self.gamma * next_max_q * float(i < len(action_list) - 1)
            
                loss = F.mse_loss(action_q, target_q)
                
                loss.backward()
                self.optimizer.step()

                # with autocast:
                #     q_values: Tensor = self.active_net(states)
                #     action_q = q_values[arange, actions]

                #     with torch.no_grad():
                #         next_q_values: Tensor = self.stable_net(next_states)
                #         next_max_q = next_q_values.max(dim=1).values
                #         target_q = rewards + self.gamma * next_max_q * float(i < len(action_list) - 1)
                
                #     loss = F.mse_loss(action_q, target_q)
                
                # grad_scaler.scale(loss).backward()
                # grad_scaler.step(self.optimizer)
                # grad_scaler.update()

                loss_item = loss.item()
                sample_loss += loss_item / horizon
                pbar.set_description(f'Training Network [Loss = {loss_item:9.6f}]')
                pbar.update()
                
                states = next_states
            
            self.update_counter += 1
            if self.update_counter % self.target_update == 0:
                self.stable_net.load_state_dict(self.active_net.state_dict())
            
            avg_loss += sample_loss / self.num_samples
        
        pbar.close()
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


