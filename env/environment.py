import torch
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F
from torch import Tensor
import numpy as np
from typing import List, Tuple, Dict, Optional, Literal
from dataclasses import dataclass
from collections import defaultdict
import os

# from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
# from overcooked_ai_py.mdp.actions import Action
# from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.utils import read_layout_dict

from utils.device import device, autocast, grad_scaler
from env.constants.actions import all_actions, NORTH, SOUTH, EAST, WEST, STAY, INTERACT, ACTION_INTERACT
from env.constants.layouts import all_layouts, LAYOUT_DIR
from env.constants.terrains import all_terrains
from env.constants.items import all_items

from env.recipes import SoupCookingTime, SoupReward
from env.state import ParallelState

class ParallelEnvironment:
    def __init__(self, layout_name: str, batch_size: int = 64, horizon: int = 400, device: torch.device = device, autocast: torch.amp.autocast = autocast):
        '''
        目前不建议使用 tutorial_1 和 tutorial_3 两个 layout
        '''
        assert layout_name not in ['tutorial_1', 'tutorial_3'], f'不支持的 layout: {layout_name}'

        self.layout_name = layout_name
        self.batch_size = batch_size
        self.horizon = horizon
        self.device = device
        self.autocast = autocast

        terrain, soup_cooking_time, soup_reward = self.load_layout(layout_name)
        self.state = ParallelState(batch_size, terrain, soup_cooking_time, soup_reward).to(device)

        self.max_ingredients = self.state.soup_cooking_time.max_ingredients
        self.timestep = 0

    def load_layout(self, layout_name: str) -> Tuple[LongTensor, SoupCookingTime, SoupReward]:
        layout_dict: Dict = read_layout_dict(layout_name)
        grid_str: str = layout_dict.get('grid')
        terrain_map = self.parse_grid(grid_str)
        soup_cooking_time, soup_reward = self.parse_soup_dicts(layout_dict)
        return terrain_map, soup_cooking_time, soup_reward

    def parse_grid(self, grid_str: str) -> LongTensor:
        rows: List[List[str]] = [list(row.strip()) for row in grid_str.strip().split('\n')]
        H, W = len(rows), len(rows[0])
        terrain_map = torch.zeros((H, W), dtype=torch.long)
        for i, row in enumerate(rows):
            for j, ch in enumerate(row):
                terrain_map[i, j] = all_terrains.get(ch)
        return terrain_map

    def parse_soup_dicts(self, layout_dict: Dict) -> Tuple[SoupCookingTime, SoupReward]:
        cook_time = layout_dict.get('cook_time', None)
        delivery_reward = layout_dict.get('delivery_reward', None)
        order_bonus = layout_dict.get('order_bonus', 2)
        
        onion_time = layout_dict.get('onion_time', None)
        tomato_time = layout_dict.get('tomato_time', None)
        recipe_time_list: List[int] = layout_dict.get('recipe_times', None)

        onion_value = layout_dict.get('onion_value', None)
        tomato_value = layout_dict.get('tomato_value', None)
        recipe_value_list: List[int] = layout_dict.get('recipe_values', None)

        start_all_orders: List[Dict[Literal['ingredients'], List[Literal['onion', 'tomato']]]] = layout_dict.get('start_all_orders')
        start_bonus_orders: List[Dict[Literal['ingredients'], List[Literal['onion', 'tomato']]]] = layout_dict.get('start_bonus_orders', None)

        all_recipes: List[Tuple[int, int]] = []
        for recipe_d in start_all_orders:
            recipe = recipe_d['ingredients']
            num_onion, num_tomato = 0, 0
            for ing in recipe:
                if ing == 'onion': num_onion += 1
                elif ing == 'tomato': num_tomato += 1
                else: raise ValueError(f'Unknown ingredient: {ing}')
            all_recipes.append((num_onion, num_tomato))

        if recipe_time_list is None or cook_time is None:
            recipe_times = None
        else:
            assert len(all_recipes) == len(recipe_time_list)
            recipe_times: Dict[Tuple[int, int], int] = dict()
            for recipe, t in zip(all_recipes, recipe_time_list):
                recipe_times[recipe] = t

        recipe_values: Dict[Tuple[int, int], int] = dict()
        if recipe_value_list is not None:
            assert len(all_recipes) == len(recipe_value_list)
            for recipe, v in zip(all_recipes, recipe_value_list):
                recipe_values[recipe] = v
        else:
            for recipe in all_recipes:
                if delivery_reward is not None:
                    recipe_values[recipe] = delivery_reward
                else:
                    assert onion_value is not None and tomato_value is not None
                    num_onion, num_tomato = recipe
                    recipe_values[recipe] = num_onion * onion_value + num_tomato * tomato_value
        
        if start_bonus_orders is not None:
            for bonus_d in start_bonus_orders:
                bonus = bonus_d['ingredients']
                num_onion, num_tomato = 0, 0
                for ing in bonus:
                    if ing == 'onion': num_onion += 1
                    elif ing == 'tomato': num_tomato += 1
                recipe_values[(num_onion, num_tomato)] *= order_bonus

        soup_cooking_time = SoupCookingTime(cook_time, recipe_times, onion_time, tomato_time)
        soup_reward = SoupReward(delivery_reward, recipe_values)
        return soup_cooking_time, soup_reward

    @torch.no_grad()
    def step(self, player1_action: LongTensor, player2_action: LongTensor) -> Tuple[bool, Tensor, Tensor, ParallelState]:
        player1_interact = (player1_action == ACTION_INTERACT)
        player2_interact = (player2_action == ACTION_INTERACT)
        player1_reward, player2_reward = self.state.interact(player1_interact, player2_interact)
        self.state.player_move(player1_action, player2_action)
        self.state.soup_cook()
        self.timestep += 1
        done = (self.timestep >= self.horizon)
        return done, player1_reward, player2_reward, self.state

