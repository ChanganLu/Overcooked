from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent

import numpy as np
import torch
torch.set_printoptions(linewidth=1000)

from env.constants.layouts import SIMPLE_O as LAYOUT
from env.constants.actions import all_actions, NORTH, SOUTH, EAST, WEST, STAY, INTERACT
from tests.extract_state import extract_state_info

layout_name = LAYOUT
mdp = OvercookedGridworld.from_layout_name(layout_name)
env = OvercookedEnv.from_mdp(mdp, horizon=400)

W, H = mdp.shape
print(H, W)

state = env.state
p1_pos, p1_ori, p1_item, p2_pos, p2_ori, p2_item, obj_tensor = extract_state_info(state, H, W)
print(state)
print(f'{p1_pos} - {p1_ori} - {p1_item} | {p2_pos} - {p2_ori} - {p2_item}')
print(torch.cat([*obj_tensor], dim=1))

done = False
total_reward = 0

steps = 0

while not done:
    action_0 = np.random.randint(0, 6)
    action_1 = np.random.randint(0, 6)
    actions = (all_actions[action_0], all_actions[action_1])
    state, reward, done, info = env.step(actions)
    p1_pos, p1_ori, p1_item, p2_pos, p2_ori, p2_item, obj_tensor = extract_state_info(state, H, W)
    print(state)
    print(f'{p1_pos} - {p1_ori} - {p1_item} | {p2_pos} - {p2_ori} - {p2_item}')
    print(torch.cat([*obj_tensor], dim=1))
    print(reward, info)
    total_reward += reward
    steps += 1
    # if steps == 200: break

print(total_reward, steps)
