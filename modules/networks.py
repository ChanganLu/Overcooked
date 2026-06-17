import torch
from torch import nn, Tensor

from env.constants.actions import all_action_ids
from env.state import TensorState
from modules.models import ResNet, MLP, StateEncoder


class ActionNet(nn.Module):
    def __init__(self, time_scale: float = 0.05, value_scale: float = 0.05, max_ingredients: float = 3.0, num_action_classes: int = len(all_action_ids)):
        super().__init__()

        self.state_encoder = StateEncoder(time_scale, value_scale, max_ingredients)
        state_encoded_channels = self.state_encoder.out_channels

        state_features = 256
        info_features = 64

        self.state_net = ResNet([state_encoded_channels, 64, 128, state_features])
        self.info_net = MLP([33, 128, info_features], use_batch_norm=False)
        self.cls_net = MLP([state_features + info_features, 512, 64, num_action_classes * num_action_classes], use_batch_norm=True)

    def forward(self, state: TensorState) -> Tensor:
        state_encoded, info_encoded = self.state_encoder(state)
        B = len(state_encoded)
        feature1: Tensor = self.state_net(state_encoded) # (B, 256, H, W)
        feature2: Tensor = self.info_net(info_encoded.unsqueeze(0)) # (1, 64)
        pooled = feature1.mean(dim=(2, 3))
        expanded = feature2.expand(B, -1)
        combined = torch.cat([pooled, expanded], dim=1)
        logits = self.cls_net(combined) # (B, 6)
        return logits

