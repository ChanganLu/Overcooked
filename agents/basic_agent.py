import torch
from torch import nn, Tensor, LongTensor
from collections import deque
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Deque, Any
import random

from utils.device import device
from env.state import TensorState
from env.environment import ParallelEnvironment
from env.constants.actions import all_action_ids


class ReplayBuffer:
    def __init__(self, capacity: int = 256):
        self.buffer: Deque[Tuple[TensorState, LongTensor, LongTensor]] = deque(maxlen=capacity)

    def push(self, states: TensorState, actions: LongTensor, rewards: LongTensor) -> None:
        horizon = len(actions)
        assert len(rewards) == horizon
        assert len(states) == horizon
        self.buffer.append((states, actions, rewards))

    def sample(self, num_samples: int) -> List[Tuple[TensorState, LongTensor, LongTensor]]:
        samples = random.sample(self.buffer, num_samples)
        return samples

class BaseAgent(ABC):
    def __init__(self, buffer_capacity: int = 1024, num_samples: int = 8, batch_size: int = 256, horizon: int = 400, num_action_classes: int = len(all_action_ids)):
        self.batch_size = batch_size
        self.horizon = horizon
        self.num_action_classes = num_action_classes
        self.num_joint_actions = num_action_classes * num_action_classes
        self.num_samples = num_samples
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

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
        states_list: List[TensorState] = []
        actions_list: List[Tensor] = []
        rewards_list: List[Tensor] = []
        avg_reward_list: List[float] = []
        done = False
        while not done:
            state = environment.state.to_tensor(environment.timestep)
            states_list.append(state.clone())
            actions = self.select_actions(state, False)
            action1 = actions // self.num_action_classes
            action2 = actions % self.num_action_classes
            done, reward1, reward2, shaped_reward1, shaped_reward2, next_state = environment.step(action1, action2)
            total_rewards = (reward1 + reward2 + shaped_reward1 + shaped_reward2).float()
            avg_reward_list.append(total_rewards.mean().item())
            actions_list.append(actions)
            rewards_list.append(total_rewards)
        self.buffer.push(TensorState.concat(states_list).cpu(), torch.cat(actions_list, dim=0).cpu(), torch.cat(rewards_list, dim=0).cpu())
        return sum(avg_reward_list)

    @abstractmethod
    def train_step(self) -> float: pass





