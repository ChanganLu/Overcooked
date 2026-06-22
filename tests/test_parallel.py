import torch
from torch import LongTensor
from utils.device import device

torch.set_printoptions(linewidth=200)

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from tests.extract_state import extract_state_info
from env.constants.terrains import TERRAIN_POT
from env.constants.items import ITEM_SOUP
from env.constants.layouts import all_available_layouts, SIMPLE_O
from env.constants.actions import all_actions
from env.state import ParallelState, TensorState
from env.environment import ParallelEnvironment

from typing import Tuple, List, Dict, Optional
from tqdm import tqdm


def compare_states(states: List[OvercookedState], parallel_state: ParallelState, step_id: int) -> Optional[str]:
    H, W = parallel_state.H, parallel_state.W
    terrains = parallel_state.terrains
    player1_positions = parallel_state.player1_position
    player1_directions = parallel_state.player1_direction
    player1_items = parallel_state.player1_item
    player2_positions = parallel_state.player2_position
    player2_directions = parallel_state.player2_direction
    player2_items = parallel_state.player2_item
    items = parallel_state.items
    soup_count_down = parallel_state.soup_count_down.clone()
    soup_count_down[(terrains != TERRAIN_POT) & (items[:, 0] == ITEM_SOUP)] = 0
    all_items = torch.cat([items, soup_count_down.unsqueeze(1)], dim=1)

    for i, state in enumerate(states):
        p1_pos, p1_ori, p1_item, p2_pos, p2_ori, p2_item, obj_tensor = extract_state_info(state, H, W)
        p_p1_pos = tuple(player1_positions[i].tolist())
        p_p1_dir = tuple(player1_directions[i].tolist())
        p_p1_item = tuple(player1_items[i].tolist())
        p_p2_pos = tuple(player2_positions[i].tolist())
        p_p2_dir = tuple(player2_directions[i].tolist())
        p_p2_item = tuple(player2_items[i].tolist())
        p_item = all_items[i]
        is_same = True
        result = f' Error at step {step_id} and env {i}:'
        if p1_pos != p_p1_pos:
            result += f'\n  Player 1 position: {p1_pos} and {p_p1_pos}'
            is_same = False
        if p1_ori != p_p1_dir:
            result += f'\n  Player 1 direction: {p1_ori} and {p_p1_dir}'
            is_same = False
        if p1_item != p_p1_item:
            result += f'\n  Player 1 item: {p1_item} and {p_p1_item}'
            is_same = False
        if p2_pos != p_p2_pos:
            result += f'\n  Player 2 position: {p2_pos} and {p_p2_pos}'
            is_same = False
        if p2_ori != p_p2_dir:
            result += f'\n  Player 2 direction: {p2_ori} and {p_p2_dir}'
            is_same = False
        if p2_item != p_p2_item:
            result += f'\n  Player 2 item: {p2_item} and {p_p2_item}'
            is_same = False
        if (obj_tensor.to(device) != p_item).any():
            result += f'\n  Items:\n{obj_tensor}\n  and\n{p_item.cpu()}'
            is_same = False
        if not is_same:
            return result

def test_layout(layout_name: str = SIMPLE_O, seed: int = 137, batch_size: int = 256, horizon: int = 400):
    generator = torch.Generator(device=device).manual_seed(seed)
    actions = torch.randint(0, len(all_actions), (horizon, batch_size, 2), generator=generator, device=device)

    envs = [OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(layout_name), horizon=horizon) for _ in range(batch_size)]
    parallel_env = ParallelEnvironment(layout_name, batch_size, horizon)

    for step_id, step_actions in enumerate(tqdm(actions, desc=f'{layout_name}')):
        states: List[OvercookedState] = []
        for env_id, (env, action) in enumerate(zip(envs, step_actions)):
            p1_action, p2_action = action.tolist()
            joint_action = (all_actions[p1_action], all_actions[p2_action])
            state, reward, done, info = env.step(joint_action)
            states.append(state)
        done, p1_rew, p2_rew, parallel_state = parallel_env.step(step_actions[:, 0], step_actions[:, 1])

        # # BEGIN DEBUG
        # error_step_id = 23
        # error_env_id = 152
        # if step_id == error_step_id - 1:
        #     print(states[error_env_id])
        # if step_id == error_step_id:
        #     p1a, p2a = step_actions[error_env_id].tolist()
        #     print(f'action: {all_actions[p1a]}, {all_actions[p2a]}')
        #     print(states[error_env_id])
        # # END DEBUG

        result = compare_states(states, parallel_state, step_id)
        if result is not None:
            print(f'Layout {layout_name}: FAIL')
            print(f'{result}')
            return False
    
    print(f'Layout {layout_name}: PASS')
    return True


# test_layout('bonus_order_test')
# exit()

for layout_name in all_available_layouts:
    same = test_layout(layout_name)
    if not same:
        exit()
