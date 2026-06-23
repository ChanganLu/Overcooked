import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple, List, Dict

from env.state import TensorState
from env.constants.items import all_items, ITEM_EMPTY
from env.constants.terrains import all_terrains


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        if in_channels == out_channels and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_channels, out_channels, 1, stride, 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, tensor: Tensor) -> Tensor:
        out1 = self.relu(self.bn1(self.conv1(tensor)))
        out2 = self.bn2(self.conv2(out1))
        out = self.relu(self.res(tensor) + out2)
        return out

class ResNet(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()
        assert len(channels) >= 2

        layers: List[nn.Module] = []
        for i in range(1, len(channels)):
            layers.append(ResBlock(channels[i - 1], channels[i]))
        self.cnn = nn.Sequential(*layers)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.cnn(tensor)

class MLP(nn.Module):
    def __init__(self, feature_dims: List[int], use_batch_norm: bool):
        super().__init__()
        assert len(feature_dims) >= 2
        
        in_features = feature_dims[0]
        out_features = feature_dims[1]

        layers: List[nn.Module] = [nn.Linear(in_features, out_features)]
        for i in range(2, len(feature_dims)):
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            in_features = out_features
            out_features = feature_dims[i]
            layers.append(nn.Linear(in_features, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.mlp(tensor)

class StateEncoder(nn.Module):
    def __init__(self, timestep_scale: float = 0.0025, time_scale: float = 0.01, value_scale: float = 0.01, max_ingredients: float = 3.0):
        super().__init__()
        self.timestep_scale = timestep_scale
        self.time_scale = time_scale
        self.value_scale = value_scale
        self.max_ingredients = max_ingredients
        self.out_channels = len(all_terrains) + 3 * (len(all_items) + 2) + 2 + 2

    def forward(self, state: TensorState) -> Tuple[Tensor, Tensor]:
        num_terrain_cls = len(all_terrains) - 2 # 排除玩家
        num_item_cls = len(all_items)

        terrains = state.terrains # (B, H, W)
        items = state.items # (B, 3, H, W)
        soup_count_down = state.soup_count_down # (B, H, W)
        player1_items = state.player1_item # (B, 3)
        player2_items = state.player2_item # (B, 3)
        player1_positions = state.player1_position # (B, 2)
        player2_positions = state.player2_position # (B, 2)
        player1_directions = state.player1_direction # (B, 2)
        player2_directions = state.player2_direction # (B, 2)

        B, H, W = terrains.shape
        device = terrains.device
        arange = torch.arange(0, B, dtype=torch.long, device=device)

        x1, y1 = player1_positions[:, 0], player1_positions[:, 1]
        x2, y2 = player2_positions[:, 0], player2_positions[:, 1]
        fx1, fy1 = x1 + player1_directions[:, 0], y1 + player1_directions[:, 1]
        fx2, fy2 = x2 + player2_directions[:, 0], y2 + player2_directions[:, 1]

        player_one_hot = torch.zeros((B, 4, H, W), dtype=torch.float, device=device) # (B, 4, H, W)
        player_one_hot[arange, 0, y1, x1] = 1.0
        player_one_hot[arange, 1, y2, x2] = 1.0
        player_one_hot[arange, 2, fy1, fx1] = 1.0
        player_one_hot[arange, 3, fy2, fx2] = 1.0

        player_items = torch.full((B, 6, H, W), fill_value=ITEM_EMPTY, dtype=torch.long, device=device)
        player_items[arange, :3, y1, x1] = player1_items
        player_items[arange, 3:, y2, x2] = player2_items

        terrain_one_hot = F.one_hot(terrains, num_terrain_cls).float().permute(0, 3, 1, 2) # (B, 7, H, W)
        items_one_hot = F.one_hot(items[:, 0], num_item_cls).float().permute(0, 3, 1, 2) # (B, 5, H, W)
        player_items_one_hot = F.one_hot(player_items[:, [0, 3]], num_item_cls).float().permute(0, 1, 4, 2, 3).reshape(B, 2 * num_item_cls, H, W) # (B, 10, H, W)
        soup_idle_one_hot = (soup_count_down < 0).float().unsqueeze(1) # (B, 1, H, W)
        soup_count_down_encoded = soup_count_down.clamp_min(0).float().unsqueeze(1) / self.time_scale # (B, 1, H, W)
        onion_and_tomato_encoded = items[:, 1:] / self.max_ingredients # (B, 2, H, W)
        player_onion_and_tomato_encoded = player_items[:, [1, 2, 4, 5]] / self.max_ingredients # (B, 4, H, W)
        encoded_2d = torch.cat([player_one_hot, terrain_one_hot, items_one_hot, player_items_one_hot, soup_idle_one_hot, soup_count_down_encoded, onion_and_tomato_encoded, player_onion_and_tomato_encoded], dim=1) # (B, 34, H, W)

        rest_timesteps = state.rest_timesteps.float().unsqueeze(1) * self.timestep_scale # (B, 1)
        times = state.times.float().flatten(1) * self.time_scale # (B, 16)
        values = state.values.float().flatten(1) * self.value_scale # (B, 16)
        # print(rest_timesteps.shape, times.shape, values.shape)
        # print(rest_timesteps.device, times.device, values.device)
        encoded_1d = torch.cat([rest_timesteps, times, values], dim=1) # (B, 33)

        return encoded_2d, encoded_1d


