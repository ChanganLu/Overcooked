import torch
from torch import nn, Tensor, LongTensor
from typing import Optional, Dict, Tuple, List, Literal


class SoupCookingTime(nn.Module):
    def __init__(self,
                 cook_time: Optional[int] = None,
                 recipe_times: Optional[Dict[Tuple[int, int], int]] = None,
                 onion_time: Optional[int] = None, tomato_time: Optional[int] = None,
                 default_time: Literal[20] = 20,
                 max_ingredients: Literal[3] = 3):
        '''
        Args:
            cook_time (Optional[int]): 统一配置的烹饪时间, 若非 None 则忽略下面所有配置
            recipe_times (Optional[Dick[Tuple[int, int], int]]): 配方的烹饪时间映射表, 键为配方的洋葱数量和番茄数量, 若配方在映射表中则忽略下面所有配置
            onion_time (Optional[int]): 配方中每个洋葱的烹饪时间
            tomato_time (Optional[int]): 配方中每个番茄的烹饪时间
            default_time (Literal[20]): 默认烹饪时间
        '''
        super().__init__()
        self.cook_time = cook_time
        if recipe_times is None or len(recipe_times) == 0:
            self.recipe_times = None
        else:
            recipes = torch.tensor(tuple(recipe_times.keys()), dtype=torch.long)
            times = torch.tensor(tuple(recipe_times.items()), dtype=torch.long)
            M = recipes.max().item() + 1
            recipes_encoded = recipes[:, 0] * M + recipes[:, 1]
            indices = torch.argsort(recipes_encoded)
            self.M = M
            self.recipes = nn.Parameter(recipes[indices], requires_grad=False)
            self.recipes_encoded = nn.Parameter(recipes_encoded[indices], requires_grad=False)
            self.recipe_times = nn.Parameter(times[indices], requires_grad=False)
        self.onion_time = onion_time
        self.tomato_time = tomato_time
        self.default_time = default_time
        self.max_ingredients = max_ingredients

    @torch.no_grad()
    def get_recipe_times(self, recipes: LongTensor) -> LongTensor:
        '''
        Args:
            recipes (LongTensor): 形状为 (B, 2), 第一列为洋葱数量, 第二列为番茄数量
        
        Returns:
            LongTensor: 形状为 (B,), 汤的烹饪时间
        '''
        device = recipes.device
        B, _ = recipes.shape # (B, 2)
        if self.cook_time is not None:
            return torch.full((B,), fill_value=self.cook_time, dtype=torch.long, device=device)
        times = torch.full((B,), fill_value=self.default_time, dtype=torch.long, device=device)
        remain_mask = torch.ones((B,), dtype=torch.bool, device=device)
        if self.recipe_times is not None:
            recipes_encoded = recipes[:, 0] * self.M + recipes[:, 1] # (B,)
            indices = torch.searchsorted(self.recipes_encoded, recipes_encoded) # (B,)
            valid = indices < len(self.recipes_encoded) # (B,)
            is_same = (recipes[valid] == self.recipes[indices[valid]]).all(dim=1)
            valid[valid.clone()] = is_same
            times[valid] = self.recipe_times[indices[valid]]
            remain_mask[valid] = False
        if not remain_mask.any().item():
            return times
        remain_recipes = recipes[remain_mask] # (R, 2)
        if self.onion_time is not None:
            if self.tomato_time is not None:
                times[remain_mask] = self.onion_time * remain_recipes[:, 0] + self.tomato_time * remain_recipes[:, 1]
            else:
                onion_only_mask = (remain_recipes[:, 1] == 0)
                remain_mask[~onion_only_mask] = False
                if onion_only_mask.any():
                    times[remain_mask] = self.onion_time * remain_recipes[onion_only_mask, 0]
        elif self.tomato_time is not None:
            tomato_only_mask = (remain_recipes[:, 0] == 0)
            remain_mask[~tomato_only_mask] = False
            if tomato_only_mask.any():
                times[remain_mask] = self.tomato_time * remain_recipes[tomato_only_mask, 1]
        return times

    def to_tensor(self) -> LongTensor:
        M = self.max_ingredients + 1
        if self.cook_time is not None: return torch.full((M, M), fill_value=self.cook_time, dtype=torch.long)
        recipe_times = torch.full((M, M), fill_value=self.default_time, dtype=torch.long)
        filled = torch.zeros((M, M), dtype=torch.bool)
        if self.recipe_times is not None:
            recipes = self.recipes.tolist()
            times = self.recipe_times.tolist()
            for (num_onion, num_tomato), t in zip(recipes, times):
                recipe_times[num_onion, num_tomato] = t
                filled[num_onion, num_tomato] = True
        if self.onion_time is not None and self.tomato_time is not None:
            onion_times = torch.arange(0, M).view(M, 1) * self.onion_time
            tomato_times = torch.arange(0, M).view(1, M) * self.tomato_time
            ingredient_times = onion_times + tomato_times
            recipe_times[~filled] = ingredient_times[~filled]
        return recipe_times
        


class SoupReward(nn.Module):
    def __init__(self,
                 delivery_reward: Optional[int] = None,
                 recipe_values: Optional[Dict[Tuple[int, int], int]] = None,
                 default_value: Literal[0] = 0,
                 max_ingredients: Literal[3] = 3):
        '''
        Args:
            delivery_reward (Optional[int]): 统一配置的奖励, 若非 None 则忽略下面所有配置
            recipe_values (Optional[Dick[Tuple[int, int], int]]): 配方的奖励映射表, 键为配方的洋葱数量和番茄数量, 所有配方 **必须** 出现在映射表内
            default_value (Literal[0]): 错误的配方没有奖励
        '''
        super().__init__()
        self.delivery_reward = delivery_reward
        if recipe_values is None or len(recipe_values) == 0:
            self.recipe_values = None
        else:
            recipes = torch.tensor(tuple(recipe_values.keys()), dtype=torch.long)
            values = torch.tensor(tuple(recipe_values.values()), dtype=torch.long)
            M = recipes.max().item() + 1
            recipes_encoded = recipes[:, 0] * M + recipes[:, 1]
            indices = torch.argsort(recipes_encoded)
            self.M = M
            self.recipes = nn.Parameter(recipes[indices], requires_grad=False)
            self.recipes_encoded = nn.Parameter(recipes_encoded[indices], requires_grad=False)
            self.recipe_values = nn.Parameter(values[indices], requires_grad=False)
        self.default_value = default_value
        self.max_ingredients = max_ingredients

    @torch.no_grad()
    def get_recipe_values(self, recipes: LongTensor) -> LongTensor:
        """
        Args:
            recipes (LongTensor): 形状 (B, 2), 第一列为洋葱数量, 第二列为番茄数量

        Returns:
            LongTensor: 形状 (B,), 汤的价值
        """
        device = recipes.device
        B = recipes.shape[0]
        if self.delivery_reward is not None:
            return torch.full((B,), fill_value=self.delivery_reward, dtype=torch.long, device=device)
        values = torch.full((B,), fill_value=self.default_value, dtype=torch.long, device=device)
        if self.recipe_values is not None:
            encoded = recipes[:, 0] * self.M + recipes[:, 1]  # (B,)
            indices = torch.searchsorted(self.recipes_encoded, encoded)  # (B,)
            valid = indices < len(self.recipes_encoded)
            is_same = (recipes[valid] == self.recipes[indices[valid]]).all(dim=1)
            valid[valid.clone()] = is_same
            values[valid] = self.recipe_values[indices[valid]]
        return values

    def to_tensor(self) -> LongTensor:
        M = self.max_ingredients + 1
        if self.delivery_reward is not None: return torch.full((M, M), fill_value=self.delivery_reward, dtype=torch.long)
        recipe_values = torch.zeros((M, M), dtype=torch.long)
        if self.recipe_values is not None:
            recipes = self.recipes.tolist()
            times = self.recipe_values.tolist()
            for (num_onion, num_tomato), t in zip(recipes, times):
                recipe_values[num_onion, num_tomato] = t
        return recipe_values

