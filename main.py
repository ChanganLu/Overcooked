import torch

from tqdm import tqdm

from utils.device import device, autocast

from env.constants.actions import all_actions
from env.constants.layouts import all_layouts, SIMPLE_O
from env.state import ParallelState, TensorState
from env.environment import ParallelEnvironment




def main():
    batch_size = 1048576
    horizon = 400
    environment = ParallelEnvironment(SIMPLE_O, batch_size, horizon=horizon)
    old_state = environment.state.to_tensor()
    # print(old_state.terrains)
    # print(old_state.player1_position.flatten().cpu())
    # print(old_state.player2_position.flatten().cpu())
    # print(old_state.player1_direction.flatten().cpu())
    # print(old_state.player2_direction.flatten().cpu())
    # return
    total_reward = torch.zeros((batch_size), dtype=torch.long, device=device)
    for i in tqdm(range(1, horizon + 1)):
        action1 = torch.randint(0, 6, (batch_size,), dtype=torch.long, device=device)
        action2 = torch.randint(0, 6, (batch_size,), dtype=torch.long, device=device)
        done, reward1, reward2, state = environment.step(action1, action2)
        new_state = state.to_tensor()
        old_state = new_state
        total_reward += reward1 + reward2
    
    print(total_reward.sum().item())


if __name__ == '__main__':
    main()
