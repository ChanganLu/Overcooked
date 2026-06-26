import torch
from torch import nn, Tensor, LongTensor, BoolTensor
from torch.utils.tensorboard import SummaryWriter

from collections import deque
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Deque, Any, Literal
import random
import math
from tqdm import tqdm

from utils.device import device
from env.state import TensorState
from env.environment import ParallelEnvironment
from env.constants.actions import all_action_ids
from env.constants.size import MAX_H, MAX_W


class ReplayBuffer:
    def __init__(self, unit: Literal['item', 'trajectory', 'batched_trajectory'] = 'item', batch_capacity: int = 256, batch_size: int = 64, horizon: int = 400):
        self.buffer_states: List[TensorState] = []
        self.buffer_actions: List[LongTensor] = []
        self.buffer_rewards: List[LongTensor] = []
        self.buffer_dones: List[BoolTensor] = []
        self.buffer_next_states: List[TensorState] = []
        self.unit = unit
        self.batch_items = batch_size * horizon
        self.capacity = batch_capacity * self.batch_items
        self.batch_size = batch_size
        self.horizon = horizon

    def __len__(self):
        return sum([actions.shape[0] for actions in self.buffer_actions])
    
    def _pop(self) -> None:
        length = len(self)
        if length > self.capacity:
            if self.unit == 'item':
                rewards = self.buffer_rewards[0].float() * 0.2
                probs = torch.softmax(rewards, dim=0)
                indices = torch.multinomial(probs, self.capacity, False)
                # indices = torch.randperm(length)[:self.capacity]
                self.buffer_states = [self.buffer_states[0][indices]]
                self.buffer_actions = [self.buffer_actions[0][indices]]
                self.buffer_rewards = [self.buffer_rewards[0][indices]]
                self.buffer_dones = [self.buffer_dones[0][indices]]
                self.buffer_next_states = [self.buffer_next_states[0][indices]]
            else: raise NotImplementedError

    
    def _concat_and_pop(self) -> None:
        if len(self.buffer_states) > 1:
            self.buffer_states = [TensorState.concat(self.buffer_states)]
            self.buffer_actions = [torch.cat(self.buffer_actions, dim=0)]
            self.buffer_rewards = [torch.cat(self.buffer_rewards, dim=0)]
            self.buffer_dones = [torch.cat(self.buffer_dones, dim=0)]
            self.buffer_next_states = [TensorState.concat(self.buffer_next_states)]
        self._pop()
        torch.cuda.empty_cache()   

    def push_batch(self, states: TensorState, actions: LongTensor, rewards: LongTensor, dones: BoolTensor, next_states: TensorState) -> None:
        self.buffer_states.append(states)
        self.buffer_actions.append(actions)
        self.buffer_rewards.append(rewards)
        self.buffer_dones.append(dones)
        self.buffer_next_states.append(next_states)
    
    def sample(self, num_samples: int):
        if self.unit == 'item':
            self._concat_and_pop()
            length = len(self)
            indices = torch.randperm(length, device=self.buffer_rewards[0].device)[:num_samples]
            return self.buffer_states[0][indices], self.buffer_actions[0][indices], self.buffer_rewards[0][indices], self.buffer_dones[0][indices], self.buffer_next_states[0][indices]
        # elif self.unit == 'trajectory':
        else: raise NotImplementedError

class BaseAgent(ABC):
    def __init__(self, buffer_capacity: int = 1024, num_samples: int = 8, batch_size: int = 256, horizon: int = 400, sample_unit: Literal['item', 'trajectory', 'batched_trajectory'] = 'item', num_action_classes: int = len(all_action_ids)):
        self.batch_size = batch_size
        self.horizon = horizon
        self.num_action_classes = num_action_classes
        self.num_joint_actions = num_action_classes * num_action_classes
        self.num_samples = num_samples
        self.buffer = ReplayBuffer(sample_unit, buffer_capacity, batch_size, horizon)

    @abstractmethod
    def select_actions(self, state: TensorState, evaluate: bool) -> LongTensor: pass

    @abstractmethod
    def train(self) -> None: pass

    @abstractmethod
    def eval(self) -> None: pass

    @abstractmethod
    def get_state_dict(self) -> Any: pass

    @abstractmethod
    def load_from_state_dict(self, state_dict) -> None: pass

    def save(self, path: str) -> None:
        state_dict = self.get_state_dict()
        torch.save(state_dict, path)
    
    def load(self, path: str) -> None:
        state_dict = torch.load(path, weights_only=False)
        self.load_from_state_dict(state_dict)

    @torch.no_grad()
    def collect_trajectory(self, environment: ParallelEnvironment) -> float:
        self.eval()
        states = environment.reset().pad_()
        avg_rewards = 0.0
        states_list: List[TensorState] = []
        actions_list: List[LongTensor] = []
        rewards_list: List[LongTensor] = []
        dones_list: List[BoolTensor] = []
        next_states_list: List[TensorState] = []
        for i in tqdm(range(1, 1 + self.horizon), desc='Collecting trajectory', dynamic_ncols=True, leave=False):
            actions = self.select_actions(states, False)
            action1 = actions // self.num_action_classes
            action2 = actions % self.num_action_classes
            done, reward1, reward2, shaped_reward1, shaped_reward2, next_states = environment.step(action1, action2)
            rewards = reward1 + reward2 + shaped_reward1 + shaped_reward2
            dones = torch.full_like(actions, fill_value=done, dtype=torch.bool)
            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            dones_list.append(dones)
            next_states_list.append(next_states.pad_())
            states = next_states
            avg_rewards += rewards.float().mean().item()
        self.buffer.push_batch(TensorState.concat(states_list), torch.cat(actions_list), torch.cat(rewards_list), torch.cat(dones_list), TensorState.concat(next_states_list))
        return avg_rewards

    @abstractmethod
    def train_step(self, writer: SummaryWriter, global_step: int) -> Tuple[float, int]: pass





