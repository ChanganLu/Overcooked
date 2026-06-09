from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent

import numpy as np

from env.constants.layouts import LARGE_ROOM as LAYOUT
from env.constants.actions import all_actions, NORTH, SOUTH, EAST, WEST, STAY, INTERACT

layout_name = LAYOUT
mdp = OvercookedGridworld.from_layout_name(layout_name)
env = OvercookedEnv.from_mdp(mdp, horizon=400)

obs = env.reset()
done = False
total_reward = 0

steps = 0

action_list = [
    ()
]

while not done:
    action_0 = np.random.randint(0, 6)
    action_1 = np.random.randint(0, 6)
    actions = (all_actions[action_0], all_actions[action_1])
    # print(actions)
    obs, reward, done, info = env.step(actions)
    # print(actions, type(obs))
    # print(obs)
    total_reward += reward
    steps += 1
    # if steps == 10: break

print(total_reward, steps)
