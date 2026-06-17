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
        self.buffer: Deque[Tuple[List[TensorState], List[LongTensor], List[LongTensor]]] = deque(maxlen=capacity)

    def push(self, states: List[TensorState], actions: List[LongTensor], rewards: List[LongTensor]) -> None:
        horizon = len(actions)
        assert len(rewards) == horizon
        assert len(states) == horizon + 1
        self.buffer.append((states, actions, rewards))

    def sample(self, num_samples: int) -> List[Tuple[List[TensorState], List[LongTensor], List[LongTensor]]]:
        samples = random.sample(self.buffer, num_samples)
        return samples

class BaseAgent(ABC):
    def __init__(self, buffer_capacity: int = 1024, num_samples: int = 8, num_action_classes: int = len(all_action_ids)):
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
        state = environment.state.to_tensor(environment.timestep)
        states_list: List[TensorState] = [state.clone().cpu()]
        actions_list: List[LongTensor] = []
        rewards_list: List[LongTensor] = []
        avg_reward_list: List[float] = []
        done = False
        while not done:
            actions = self.select_actions(state, False)
            action1 = actions // self.num_action_classes
            action2 = actions % self.num_action_classes
            done, reward1, reward2, next_state = environment.step(action1, action2)
            total_rewards = reward1 + reward2
            avg_reward_list.append(total_rewards.float().mean().item())
            state = next_state.to_tensor(environment.timestep)
            state_cpu = state.clone().cpu()
            actions_cpu = actions.cpu()
            rewards_cpu = total_rewards.cpu()
            states_list.append(state_cpu)
            actions_list.append(actions_cpu)
            rewards_list.append(rewards_cpu)
        self.buffer.push(states_list, actions_list, rewards_list)
        return sum(avg_reward_list) / len(avg_reward_list)

    @abstractmethod
    def train_step(self) -> float: pass





