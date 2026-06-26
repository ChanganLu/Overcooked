import torch
import numpy as np

from tqdm import tqdm
from datetime import datetime
import os

from utils.device import device, autocast

from env.constants.actions import all_actions
from env.constants.layouts import all_prefered_layouts as layouts_to_use # all_layouts, all_available_layouts, SIMPLE_O
from env.state import ParallelState, TensorState
from env.environment import ParallelEnvironment

from agents.basic_agent import BaseAgent, SummaryWriter
from agents.DQN import DQNAgent

# np.random.seed(42)


def train_agent(agent: BaseAgent, writer: SummaryWriter, num_episodes: int = 10000, save_dir: str = './checkpoint/default') -> BaseAgent:
    num_avaliable_layouts = len(layouts_to_use)
    agent.save(os.path.join(save_dir, f'0.pth'))
    pbar = tqdm(total=num_episodes, dynamic_ncols=True, desc=f'Training Agent [reward = ???.??????] [loss = ???.??????]')
    loss = 0.0
    reward = 0.0
    global_step = 0
    for i in range(1, 1 + num_episodes):
        try:
            layout = layouts_to_use[np.random.randint(0, num_avaliable_layouts)]
            pbar.set_description(desc=f'Training Agent (layout: {layout}) [reward = {reward:10.6f}] [loss = {loss:10.6f}]')
            # layout = 'limited_0'
            # layout = 'you_shall_not_pass'
            environment = ParallelEnvironment(layout, batch_size=agent.batch_size, horizon=agent.horizon, enable_reward_shaping=True)
            reward = agent.collect_trajectory(environment)
            writer.add_scalar('avg reward', reward, global_step)
            pbar.set_description(desc=f'Training Agent (layout: {layout}) [reward = {reward:10.6f}] [loss = {loss:10.6f}]')
            loss, global_step = agent.train_step(writer, global_step)
            pbar.set_description(desc=f'Training Agent (layout: {layout}) [reward = {reward:10.6f}] [loss = {loss:10.6f}]')
            pbar.update(1)
            if i % 50 == 0:
                agent.save(os.path.join(save_dir, f'{i}.pth'))
        except KeyboardInterrupt as e:
            agent.save(os.path.join(save_dir, f'{i}.pth'))
            pbar.close()
            writer.close()
            print('Training interruped by user.')
            raise e
            return agent
    pbar.close()
    return agent


def main():
    # now_str = datetime.now().strftime("[%Y-%m-%d]-[%H-%M-%S]")
    # save_dir = os.path.join('./checkpoint', now_str)
    # save_dir = './checkpoint/default'
    out_dir = 'checkpoints'
    config_name = 'pre_train'
    save_dir = os.path.join(out_dir, config_name, 'agent')
    log_dir = os.path.join(out_dir, config_name, 'logs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    agent = DQNAgent(buffer_capacity=8)
    agent.load('checkpoints/1/agent/50.pth')
    writer = SummaryWriter(log_dir)
    train_agent(agent, writer, 2500, save_dir)
    writer.close()


if __name__ == '__main__':
    main()
