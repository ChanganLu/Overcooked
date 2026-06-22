import torch
from utils.device import device
from env.environment import ParallelEnvironment
from env.constants.layouts import ASYMMETRIC_ADVANTAGES_TOMATO

env = ParallelEnvironment(ASYMMETRIC_ADVANTAGES_TOMATO)
soup_reward = env.state.soup_reward

recipes = torch.tensor([
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 0],
    [1, 1],
    [1, 2],
    [2, 0],
    [2, 1],
    [3, 0]
], device=device, dtype=torch.long)
values = soup_reward.get_recipe_values(recipes)
print(torch.cat([recipes, values.unsqueeze(1)], dim=1))
