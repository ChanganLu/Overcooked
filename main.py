import torch
import numpy as np

from tqdm import tqdm
from datetime import datetime
import os

from utils.device import device, autocast

from env.constants.actions import all_actions
from env.constants.layouts import all_layouts, all_available_layouts, SIMPLE_O
from env.state import ParallelState, TensorState
from env.environment import ParallelEnvironment

from agents.basic_agent import BaseAgent
from agents.DQN import DQNAgent

# np.random.seed(42)


def train_agent(agent: BaseAgent,num_episodes: int = 1000, save_dir: str = './checkpoint/default') -> BaseAgent:
    os.makedirs(save_dir, exist_ok=True)
    num_avaliable_layouts = len(all_available_layouts)
    agent.save(os.path.join(save_dir, f'0.pth'))
    pbar = tqdm(total=num_episodes, dynamic_ncols=True, desc=f'Training Agent [reward = ???.????????] [loss = ???.????????]')
    loss = 0.0
    for i in range(1, 1 + num_episodes):
        try:
            layout = all_available_layouts[np.random.randint(0, num_avaliable_layouts)]
            # layout = 'limited_0'
            # layout = 'you_shall_not_pass'
            environment = ParallelEnvironment(layout, batch_size=agent.batch_size, horizon=agent.horizon, enable_reward_shaping=True)
            reward = agent.collect_trajectory(environment)
            loss = agent.train_step()
            pbar.set_description(desc=f'Training Agent (layout: {layout}) [reward = {reward:12.8f}] [loss = {loss:12.8f}]')
            pbar.update(1)
            if i % 50 == 0:
                agent.save(os.path.join(save_dir, f'{i}.pth'))
        except KeyboardInterrupt:
            agent.save(os.path.join(save_dir, f'{i}.pth'))
            pbar.close()
            print('Training interruped by user.')
            return agent
    pbar.close()
    return agent


def main():
    # now_str = datetime.now().strftime("[%Y-%m-%d]-[%H-%M-%S]")
    # save_dir = os.path.join('./checkpoint', now_str)
    # save_dir = './checkpoint/default'
    save_dir = './checkpoint/pre_train'
    agent = DQNAgent(num_samples=4)
    # agent.load('./checkpoint/default/800.pth')
    # agent.epsilon = 0.8
    # agent.optimizer = torch.optim.AdamW(agent.active_net.parameters(), lr=5e-4, weight_decay=1e-3)
    train_agent(agent, save_dir=save_dir)


if __name__ == '__main__':
    main()
