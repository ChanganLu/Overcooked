import torch
from torch import nn, Tensor, LongTensor, BoolTensor
from typing import List, Tuple, Dict, Optional, Literal, Self
from dataclasses import dataclass

from env.recipes import SoupCookingTime, SoupReward


from env.constants.terrains import TERRAIN_PLAYER_1, TERRAIN_PLAYER_2, TERRAIN_EMPTY, TERRAIN_COUNTER, TERRAIN_ONION_DISP, TERRAIN_TOMATO_DISP, TERRAIN_DISH_DISP, TERRAIN_POT, TERRAIN_SERVE
from env.constants.items import ITEM_EMPTY, ITEM_ONION, ITEM_TOMATO, ITEM_DISH, ITEM_SOUP
from env.constants.actions import ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST, ACTION_STAY, ACTION_INTERACT # 0, 1, 2, 3, 4, 5
from env.constants.actions import NORTH, SOUTH, EAST, WEST, STAY # (0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)



class TensorState:
    def __init__(self,
                 terrains: LongTensor, items: LongTensor, soup_count_down: LongTensor,
                 player1_item: LongTensor, player2_item: LongTensor, player1_position: LongTensor, player2_position: LongTensor, player1_direction: LongTensor, player2_direction: LongTensor,
                 rest_timesteps: int, times: LongTensor, values: LongTensor):
        device = terrains.device
        self.terrains = terrains.clone().to(device)
        self.items = items.clone().to(device)
        self.soup_count_down = soup_count_down.clone().to(device)
        self.player1_item = player1_item.clone().to(device)
        self.player2_item = player2_item.clone().to(device)
        self.player1_position = player1_position.clone().to(device)
        self.player2_position = player2_position.clone().to(device)
        self.player1_direction = player1_direction.clone().to(device)
        self.player2_direction = player2_direction.clone().to(device)
        self.rest_timesteps = rest_timesteps
        self.times = times.clone().to(device)
        self.values = values.clone().to(device)

    def to(self, device: torch.device) -> Self:
        self.terrains = self.terrains.to(device)
        self.items = self.items.to(device)
        self.soup_count_down = self.soup_count_down.to(device)
        self.player1_item = self.player1_item.to(device)
        self.player2_item = self.player2_item.to(device)
        self.player1_position = self.player1_position.to(device)
        self.player2_position = self.player2_position.to(device)
        self.player1_direction = self.player1_direction.to(device)
        self.player2_direction = self.player2_direction.to(device)
        self.times = self.times.to(device)
        self.values = self.values.to(device)
        return self

    def cpu(self) -> Self: return self.to(torch.device('cpu'))
    def cuda(self) -> Self: return self.to(torch.device('cuda'))

    def clone(self) -> 'TensorState': return TensorState(self.terrains, self.items, self.soup_count_down, self.player1_item, self.player2_item, self.player1_position, self.player2_position, self.player1_direction, self.player2_direction, self.rest_timesteps, self.times, self.values)


class ParallelState(nn.Module):
    def __init__(self, batch_size: int, terrain_map: LongTensor, soup_cooking_time: SoupCookingTime, soup_reward: SoupReward, enable_reward_shaping: bool = False):
        super().__init__()
        self.enable_reward_shaping = enable_reward_shaping

        H, W = terrain_map.shape
        B = batch_size

        self.H, self.W = H, W
        self.B = B

        x1, y1 = torch.where(terrain_map == TERRAIN_PLAYER_1)
        x2, y2 = torch.where(terrain_map == TERRAIN_PLAYER_2)
        player1_position = [y1.item(), x1.item()]
        player2_position = [y2.item(), x2.item()]

        terrain = terrain_map.clone()
        terrain[player1_position[1], player1_position[0]] = TERRAIN_EMPTY
        terrain[player2_position[1], player2_position[0]] = TERRAIN_EMPTY

        self.terrains           = nn.Parameter(terrain.unsqueeze(0).expand(B, -1, -1), requires_grad=False)
        '''(B, H, W)'''
        self.items              = nn.Parameter(torch.zeros((B, 3, H, W), dtype=torch.long), requires_grad=False)
        '''(B, 3, H, W), 分别为物品 ID, 洋葱数量, 番茄数量'''
        self.soup_count_down    = nn.Parameter(torch.full_like(self.terrains, fill_value=-1), requires_grad=False)
        '''(B, H, W), -1 表示未开始烹饪, 0 表示完成烹饪, 正整数表示烹饪剩余时间'''
        self.player1_item       = nn.Parameter(torch.zeros((B, 3), dtype=torch.long), requires_grad=False)
        '''(B, 3)'''
        self.player2_item       = nn.Parameter(torch.zeros((B, 3), dtype=torch.long), requires_grad=False)
        '''(B, 3)'''
        self.player1_position   = nn.Parameter(torch.tensor([player1_position], dtype=torch.long).expand(B, -1), requires_grad=False)
        '''(B, 2)'''
        self.player2_position   = nn.Parameter(torch.tensor([player2_position], dtype=torch.long).expand(B, -1), requires_grad=False)
        '''(B, 2)'''
        self.player1_direction  = nn.Parameter(torch.tensor([NORTH], dtype=torch.long).expand(B, -1), requires_grad=False)
        '''(B, 2)'''
        self.player2_direction  = nn.Parameter(torch.tensor([NORTH], dtype=torch.long).expand(B, -1), requires_grad=False)
        '''(B, 2)'''
        self.action_directions  = nn.Parameter(torch.tensor([NORTH, SOUTH, EAST, WEST, STAY, STAY], dtype=torch.long), requires_grad=False)
        '''(6, 2)'''

        self.soup_cooking_time = soup_cooking_time
        self.soup_reward = soup_reward

    @torch.no_grad()
    def player_move(self, player1_action: LongTensor, player2_action: LongTensor) -> None:
        arange = torch.arange(0, len(player1_action), dtype=torch.long, device=player1_action.device)
        player1_move_mask = player1_action <= ACTION_WEST
        player2_move_mask = player2_action <= ACTION_WEST

        player1_step = self.action_directions[player1_action] # (B, 2)
        player2_step = self.action_directions[player2_action] # (B, 2)

        player1_new_position = self.player1_position + player1_step # (B, 2)
        player2_new_position = self.player2_position + player2_step # (B, 2)

        # 检查是否碰到障碍
        player1_can_not_enter_mask = self.terrains[arange, player1_new_position[:, 1], player1_new_position[:, 0]] != TERRAIN_EMPTY
        player2_can_not_enter_mask = self.terrains[arange, player2_new_position[:, 1], player2_new_position[:, 0]] != TERRAIN_EMPTY

        # 阻挡碰到障碍物的玩家
        player1_new_position[player1_can_not_enter_mask] = self.player1_position[player1_can_not_enter_mask]
        player2_new_position[player2_can_not_enter_mask] = self.player2_position[player2_can_not_enter_mask]

        # 检查玩家是否相撞(到达同一位置或交换位置)
        player_collision_mask = (player1_new_position == player2_new_position).all(dim=1)
        player_swap_mask = torch.logical_and((player1_new_position == self.player2_position).all(dim=1), (player2_new_position == self.player1_position).all(dim=1))
        player_collision_mask.logical_or_(player_swap_mask)
        player_not_collision_mask = player_collision_mask.logical_not_()

        # 更新玩家位置和方向
        self.player1_position[player_not_collision_mask] = player1_new_position[player_not_collision_mask]
        self.player1_direction[player1_move_mask] = player1_step[player1_move_mask]
        self.player2_position[player_not_collision_mask] = player2_new_position[player_not_collision_mask]
        self.player2_direction[player2_move_mask] = player2_step[player2_move_mask]

    @torch.no_grad()
    def interact(self, player1_interact: BoolTensor, player2_interact: BoolTensor) -> Tuple[LongTensor, LongTensor]:
        device = player1_interact.device
        B = len(player1_interact)
        # arange = torch.arange(0, B, dtype=torch.long, device=device)
        # 先处理玩家 1 的交互
        player1_reward = torch.zeros((B,), dtype=torch.long, device=device)
        # player1_reward_shaping = torch.zeros((B,), dtype=torch.long, device=device)
        if player1_interact.any():
            player_front = self.player1_position[player1_interact] + self.player1_direction[player1_interact] # (I, 2)
            player_front_terrain = self.terrains[player1_interact, player_front[:, 1], player_front[:, 0]] # (I,)
            player_front_items = self.items[player1_interact, :, player_front[:, 1], player_front[:, 0]] # (I, 3)
            # print(f'player1 front items shape {player_front_items.shape}')
            player_front_soup_count_down = self.soup_count_down[player1_interact, player_front[:, 1], player_front[:, 0]] # (I,)
            player_items = self.player1_item[player1_interact] # (I, 3)
            player_rewards = torch.zeros_like(player_front_terrain) # (I,)
            # 1. 玩家空手
            player_empty_mask = (player_items[:, 0] == ITEM_EMPTY) # (I,)
            if player_empty_mask.any():
                terrain = player_front_terrain[player_empty_mask] # (E,)
                items = player_front_items[player_empty_mask] # (E, 3)
                soup_count_down = player_front_soup_count_down[player_empty_mask] # (E,)
                holding_items = player_items[player_empty_mask] # (E, 3)
                E = len(terrain)
                rewards = torch.zeros_like(terrain)
                # 1.1 从柜台上拿物品
                counter_mask = (terrain == TERRAIN_COUNTER) # (E,)
                counter_dish_mask = (items[:, 0] == ITEM_DISH) # (E,)
                if counter_mask.any():
                    holding_items[counter_mask] = items[counter_mask]
                    items[counter_mask] = ITEM_EMPTY
                if self.enable_reward_shaping:
                    no_dish_on_counter_mask = (self.items[player1_interact, 0][player_empty_mask].sum(dim=(1, 2)) == 0) # (E,)
                    dish_pickup_useful_mask = torch.logical_and(torch.logical_and(counter_mask, counter_dish_mask), no_dish_on_counter_mask) # (E,)
                    if dish_pickup_useful_mask.any():
                        rewards[dish_pickup_useful_mask] = 3
                # 1.2 从供应器拿物品
                # a. 洋葱
                onion_mask = (terrain == TERRAIN_ONION_DISP) # (E,)
                if onion_mask.any():
                    holding_items[onion_mask, 0] = ITEM_ONION
                # b. 番茄
                tomato_mask = (terrain == TERRAIN_TOMATO_DISP) # (E,)
                if tomato_mask.any():
                    holding_items[tomato_mask, 0] = ITEM_TOMATO
                # c. 盘子
                dish_mask = (terrain == TERRAIN_DISH_DISP) # (E,)
                if dish_mask.any():
                    holding_items[dish_mask, 0] = ITEM_DISH
                    if self.enable_reward_shaping:
                        dish_pickup_useful_mask = torch.logical_and(dish_mask, no_dish_on_counter_mask) # (E,)
                        if dish_pickup_useful_mask.any():
                            rewards[dish_pickup_useful_mask] = 3
                # 1.3 与锅交互开始烹饪
                pot_mask = (terrain == TERRAIN_POT) # (E,)
                soup_idle_mask = (soup_count_down == -1) # (E,)
                soup_not_empty_mask = (items[:, 1:] > 0).any(dim=1) # (E,)
                can_begin_cooking_mask = torch.logical_and(pot_mask, torch.logical_and(soup_idle_mask, soup_not_empty_mask)) # (E,)
                if can_begin_cooking_mask.any():
                    recipes = items[can_begin_cooking_mask, 1:]
                    cooking_times = self.soup_cooking_time.get_recipe_times(recipes)
                    soup_count_down[can_begin_cooking_mask] = cooking_times
                # 写回临时状态
                player_front_items[player_empty_mask] = items
                player_front_soup_count_down[player_empty_mask] = soup_count_down
                player_items[player_empty_mask] = holding_items
                player_rewards[player_empty_mask] = rewards
            # 2. 玩家手持物品
            player_holding_mask = torch.logical_not(player_empty_mask) # (I,)
            if player_holding_mask.any():
                terrain = player_front_terrain[player_holding_mask] # (H,)
                items = player_front_items[player_holding_mask] # (H, 3)
                soup_count_down = player_front_soup_count_down[player_holding_mask] # (H,)
                holding_items = player_items[player_holding_mask] # (H, 3)
                H = len(terrain)
                rewards = torch.zeros_like(terrain)
                # 2.1 向空柜台上放置物品
                counter_mask = (terrain == TERRAIN_COUNTER) # (H,)
                counter_empty_mask = (items[:, 0] == ITEM_EMPTY) # (H,)
                can_place_item_mask = torch.logical_and(counter_mask, counter_empty_mask) # (H,)
                if can_place_item_mask.any():
                    items[can_place_item_mask] = holding_items[can_place_item_mask]
                    holding_items[can_place_item_mask] = ITEM_EMPTY
                # 2.2 与锅交互
                pot_mask = (terrain == TERRAIN_POT) # (H,)
                # a. 向锅中添加原料
                not_full_mask = torch.logical_and(soup_count_down == -1, items[:, 1:].sum(dim=1) < self.soup_cooking_time.max_ingredients) # (H,)
                can_add_ingredient_mask = torch.logical_and(pot_mask, not_full_mask) # (H,)
                # a.1 添加洋葱
                player_onion_mask = (holding_items[:, 0] == ITEM_ONION) # (H,)
                can_add_onion_mask = torch.logical_and(can_add_ingredient_mask, player_onion_mask) # (H,)
                if can_add_onion_mask.any():
                    holding_items[can_add_onion_mask] = ITEM_EMPTY
                    items[can_add_onion_mask, 0] = ITEM_SOUP
                    items[can_add_onion_mask, 1] += 1
                    if self.enable_reward_shaping:
                        rewards[can_add_onion_mask] = 3
                # a.2 添加番茄
                player_tomato_mask = (holding_items[:, 0] == ITEM_TOMATO) # (H,)
                can_add_tomato_mask = torch.logical_and(can_add_ingredient_mask, player_tomato_mask) # (H,)
                if can_add_tomato_mask.any():
                    holding_items[can_add_tomato_mask] = ITEM_EMPTY
                    items[can_add_tomato_mask, 0] = ITEM_SOUP
                    items[can_add_tomato_mask, 2] += 1
                    if self.enable_reward_shaping:
                        rewards[can_add_tomato_mask] = 3
                # b. 盛汤
                holding_dish_mask = (holding_items[:, 0] == ITEM_DISH) # (H,)
                soup_finish_mask = (soup_count_down == 0) # (H,)
                can_get_soup_mask = torch.logical_and(pot_mask, torch.logical_and(holding_dish_mask, soup_finish_mask)) # (H,)
                if can_get_soup_mask.any():
                    holding_items[can_get_soup_mask] = items[can_get_soup_mask]
                    items[can_get_soup_mask] = ITEM_EMPTY
                    soup_count_down[can_get_soup_mask] = -1
                    if self.enable_reward_shaping:
                        rewards[can_get_soup_mask] = 5
                # 2.3 与上菜点交互
                serve_mask = (terrain == TERRAIN_SERVE) # (H,)
                holding_soup_mask = (holding_items[:, 0] == ITEM_SOUP) # (H,)
                can_serve_mask = torch.logical_and(serve_mask, holding_soup_mask) # (H,)
                if can_serve_mask.any():
                    holding_items[can_serve_mask] = ITEM_EMPTY
                    recipes = items[can_serve_mask, 1:]
                    rewards[can_serve_mask] = self.soup_reward.get_recipe_values(recipes)
                # 写回临时状态
                player_front_items[player_holding_mask] = items
                player_front_soup_count_down[player_holding_mask] = soup_count_down
                player_items[player_holding_mask] = holding_items
                player_rewards[player_holding_mask] = rewards
            # 写回全局状态
            self.items[player1_interact, :, player_front[:, 1], player_front[:, 0]] = player_front_items
            self.soup_count_down[player1_interact, player_front[:, 1], player_front[:, 0]] = player_front_soup_count_down
            self.player1_item[player1_interact] = player_items
            player1_reward[player1_interact] = player_rewards
        # 再处理玩家 2 的交互
        player2_reward = torch.zeros((B,), dtype=torch.long, device=device)
        # player2_reward_shaping = torch.zeros((B,), dtype=torch.long, device=device)
        if player2_interact.any():
            player_front = self.player2_position[player2_interact] + self.player2_direction[player2_interact] # (I, 2)
            player_front_terrain = self.terrains[player2_interact, player_front[:, 1], player_front[:, 0]] # (I,)
            player_front_items = self.items[player2_interact, :, player_front[:, 1], player_front[:, 0]] # (I, 3)
            # print(f'player2 front items shape {player_front_items.shape}')
            player_front_soup_count_down = self.soup_count_down[player2_interact, player_front[:, 1], player_front[:, 0]] # (I,)
            player_items = self.player2_item[player2_interact] # (I, 3)
            player_rewards = torch.zeros_like(player_front_terrain) # (I,)
            # 1. 玩家空手
            player_empty_mask = (player_items[:, 0] == ITEM_EMPTY) # (I,)
            if player_empty_mask.any():
                terrain = player_front_terrain[player_empty_mask] # (E,)
                items = player_front_items[player_empty_mask] # (E, 3)
                soup_count_down = player_front_soup_count_down[player_empty_mask] # (E,)
                holding_items = player_items[player_empty_mask] # (E, 3)
                E = len(terrain)
                rewards = torch.zeros_like(terrain)
                # 1.1 从柜台上拿物品
                counter_mask = (terrain == TERRAIN_COUNTER) # (E,)
                counter_dish_mask = (items[:, 0] == ITEM_DISH) # (E,)
                if counter_mask.any():
                    holding_items[counter_mask] = items[counter_mask]
                    items[counter_mask] = ITEM_EMPTY
                if self.enable_reward_shaping:
                    no_dish_on_counter_mask = (self.items[player2_interact, 0][player_empty_mask].sum(dim=(1, 2)) == 0) # (E,)
                    dish_pickup_useful_mask = torch.logical_and(torch.logical_and(counter_mask, counter_dish_mask), no_dish_on_counter_mask) # (E,)
                    if dish_pickup_useful_mask.any():
                        rewards[dish_pickup_useful_mask] = 3
                # 1.2 从供应器拿物品
                # a. 洋葱
                onion_mask = (terrain == TERRAIN_ONION_DISP) # (E,)
                if onion_mask.any():
                    holding_items[onion_mask, 0] = ITEM_ONION
                # b. 番茄
                tomato_mask = (terrain == TERRAIN_TOMATO_DISP) # (E,)
                if tomato_mask.any():
                    holding_items[tomato_mask, 0] = ITEM_TOMATO
                # c. 盘子
                dish_mask = (terrain == TERRAIN_DISH_DISP) # (E,)
                if dish_mask.any():
                    holding_items[dish_mask, 0] = ITEM_DISH
                    if self.enable_reward_shaping:
                        dish_pickup_useful_mask = torch.logical_and(dish_mask, no_dish_on_counter_mask) # (E,)
                        if dish_pickup_useful_mask.any():
                            rewards[dish_pickup_useful_mask] = 3
                # 1.3 与锅交互开始烹饪
                pot_mask = (terrain == TERRAIN_POT) # (E,)
                soup_idle_mask = (soup_count_down == -1) # (E,)
                soup_not_empty_mask = (items[:, 1:] > 0).any(dim=1) # (E,)
                can_begin_cooking_mask = torch.logical_and(pot_mask, torch.logical_and(soup_idle_mask, soup_not_empty_mask)) # (E,)
                if can_begin_cooking_mask.any():
                    recipes = items[can_begin_cooking_mask, 1:]
                    cooking_times = self.soup_cooking_time.get_recipe_times(recipes)
                    soup_count_down[can_begin_cooking_mask] = cooking_times
                # 写回临时状态
                player_front_items[player_empty_mask] = items
                player_front_soup_count_down[player_empty_mask] = soup_count_down
                player_items[player_empty_mask] = holding_items
                player_rewards[player_empty_mask] = rewards
            # 2. 玩家手持物品
            player_holding_mask = torch.logical_not(player_empty_mask) # (I,)
            if player_holding_mask.any():
                terrain = player_front_terrain[player_holding_mask] # (H,)
                items = player_front_items[player_holding_mask] # (H, 3)
                soup_count_down = player_front_soup_count_down[player_holding_mask] # (H,)
                holding_items = player_items[player_holding_mask] # (H, 3)
                H = len(terrain)
                rewards = torch.zeros_like(terrain)
                # 2.1 向空柜台上放置物品
                counter_mask = (terrain == TERRAIN_COUNTER) # (H,)
                counter_empty_mask = (items[:, 0] == ITEM_EMPTY) # (H,)
                can_place_item_mask = torch.logical_and(counter_mask, counter_empty_mask) # (H,)
                if can_place_item_mask.any():
                    items[can_place_item_mask] = holding_items[can_place_item_mask]
                    holding_items[can_place_item_mask] = ITEM_EMPTY
                # 2.2 与锅交互
                pot_mask = (terrain == TERRAIN_POT) # (H,)
                # a. 向锅中添加原料
                not_full_mask = torch.logical_and(soup_count_down == -1, items[:, 1:].sum(dim=1) < self.soup_cooking_time.max_ingredients) # (H,)
                can_add_ingredient_mask = torch.logical_and(pot_mask, not_full_mask) # (H,)
                # a.1 添加洋葱
                player_onion_mask = (holding_items[:, 0] == ITEM_ONION) # (H,)
                can_add_onion_mask = torch.logical_and(can_add_ingredient_mask, player_onion_mask) # (H,)
                if can_add_onion_mask.any():
                    holding_items[can_add_onion_mask] = ITEM_EMPTY
                    items[can_add_onion_mask, 0] = ITEM_SOUP
                    items[can_add_onion_mask, 1] += 1
                    if self.enable_reward_shaping:
                        rewards[can_add_onion_mask] = 3
                # a.2 添加番茄
                player_tomato_mask = (holding_items[:, 0] == ITEM_TOMATO) # (H,)
                can_add_tomato_mask = torch.logical_and(can_add_ingredient_mask, player_tomato_mask) # (H,)
                if can_add_tomato_mask.any():
                    holding_items[can_add_tomato_mask] = ITEM_EMPTY
                    items[can_add_tomato_mask, 0] = ITEM_SOUP
                    items[can_add_tomato_mask, 2] += 1
                    if self.enable_reward_shaping:
                        rewards[can_add_tomato_mask] = 3
                # b. 盛汤
                holding_dish_mask = (holding_items[:, 0] == ITEM_DISH) # (H,)
                soup_finish_mask = (soup_count_down == 0) # (H,)
                can_get_soup_mask = torch.logical_and(pot_mask, torch.logical_and(holding_dish_mask, soup_finish_mask)) # (H,)
                if can_get_soup_mask.any():
                    holding_items[can_get_soup_mask] = items[can_get_soup_mask]
                    items[can_get_soup_mask] = ITEM_EMPTY
                    soup_count_down[can_get_soup_mask] = -1
                    if self.enable_reward_shaping:
                        rewards[can_get_soup_mask] = 5
                # 2.3 与上菜点交互
                serve_mask = (terrain == TERRAIN_SERVE) # (H,)
                holding_soup_mask = (holding_items[:, 0] == ITEM_SOUP) # (H,)
                can_serve_mask = torch.logical_and(serve_mask, holding_soup_mask) # (H,)
                if can_serve_mask.any():
                    holding_items[can_serve_mask] = ITEM_EMPTY
                    recipes = items[can_serve_mask, 1:]
                    rewards[can_serve_mask] = self.soup_reward.get_recipe_values(recipes)
                # 写回临时状态
                player_front_items[player_holding_mask] = items
                player_front_soup_count_down[player_holding_mask] = soup_count_down
                player_items[player_holding_mask] = holding_items
                player_rewards[player_holding_mask] = rewards
            # 写回全局状态
            self.items[player2_interact, :, player_front[:, 1], player_front[:, 0]] = player_front_items
            self.soup_count_down[player2_interact, player_front[:, 1], player_front[:, 0]] = player_front_soup_count_down
            self.player2_item[player2_interact] = player_items
            player2_reward[player2_interact] = player_rewards
        return player1_reward, player2_reward

    @torch.no_grad()
    def soup_cook(self) -> None:
        cooking_mask = (self.soup_count_down > 0)
        self.soup_count_down[cooking_mask] -= 1

    def to_tensor(self, rest_timesteps: int) -> TensorState:
        return TensorState(
            self.terrains, self.items, self.soup_count_down,
            self.player1_item, self.player2_item, self.player1_position, self.player2_position, self.player1_direction, self.player2_direction,
            rest_timesteps, self.soup_cooking_time.to_tensor(), self.soup_reward.to_tensor()
        )

