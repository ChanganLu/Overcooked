import torch
from torch import LongTensor
from utils.device import device

torch.set_printoptions(linewidth=200)

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from tests.extract_state import extract_state_info
from env.constants.terrains import TERRAIN_POT
from env.constants.items import ITEM_SOUP
from env.constants.layouts import all_available_layouts, SIMPLE_O, YOU_SHALL_NOT_PASS
from env.constants.actions import all_actions
from env.state import ParallelState, TensorState
from env.environment import ParallelEnvironment

from typing import Tuple, List, Dict, Optional, Union
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

def test_layout_transition(layout_name: str = SIMPLE_O, seed: int = 137, batch_size: int = 256, horizon: int = 400):
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
        done, p1_rew, p2_rew, p1_s_rew, p2_s_rew, parallel_state = parallel_env.step(step_actions[:, 0], step_actions[:, 1])

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

def test_layout_reward(layout_name: str = SIMPLE_O, seed: int = 137, batch_size: int = 256, horizon: int = 400):
    generator = torch.Generator(device=device).manual_seed(seed)
    actions = torch.randint(0, len(all_actions), (horizon, batch_size, 2), generator=generator, device=device)

    envs = [OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(layout_name), horizon=horizon) for _ in range(batch_size)]
    parallel_env = ParallelEnvironment(layout_name, batch_size, horizon, enable_reward_shaping=True)
    total_reward = 0
    total_shaped_reward = 0

    for step_id, step_actions in enumerate(tqdm(actions, desc=f'{layout_name}')):
        rewards: List[List[int]] = []
        shaped_rewards: List[List[int]] = []
        states: List[OvercookedState] = []
        for env_id, (env, action) in enumerate(zip(envs, step_actions)):
            p1_action, p2_action = action.tolist()
            joint_action = (all_actions[p1_action], all_actions[p2_action])
            state, reward, done, info = env.step(joint_action)
            states.append(state)
            reward = info["sparse_r_by_agent"]
            shaped_reward = info["shaped_r_by_agent"]
            rewards.append(reward)
            shaped_rewards.append(shaped_reward)
        done, p_p1_rew, p_p2_rew, p_p1_s_rew, p_p2_s_rew, parallel_state = parallel_env.step(step_actions[:, 0], step_actions[:, 1])
        rewards_tensor: LongTensor = torch.tensor(rewards, dtype=torch.long, device=device)
        shaped_rewards_tensor: LongTensor = torch.tensor(shaped_rewards, dtype=torch.long, device=device)
        p1_rew = rewards_tensor[:, 0]
        p2_rew = rewards_tensor[:, 1]
        p1_s_rew = shaped_rewards_tensor[:, 0]
        p2_s_rew = shaped_rewards_tensor[:, 1]

        # # BEGIN DEBUG
        # error_step_id = 353
        # error_env_id = 161
        # if error_step_id - 1 <= step_id <= error_step_id - 1:
        #     print(states[error_env_id])
        # if step_id == error_step_id:
        #     p1a, p2a = step_actions[error_env_id].tolist()
        #     print(f'action: {all_actions[p1a]}, {all_actions[p2a]}')
        #     print(states[error_env_id])
        #     print(f'GT reward = {rewards[error_env_id]} | GT shaped reward = {shaped_rewards[error_env_id]}')
        #     print(f'Parallel reward = {[p_p1_rew[error_env_id].item(), p_p2_rew[error_env_id].item()]} | Parallel shaped reward = {[p_p1_s_rew[error_env_id].item(), p_p2_s_rew[error_env_id].item()]}')
        # # END DEBUG

        result = f' Error at step {step_id}'
        is_same = True
        if (p1_rew != p_p1_rew).any():
            neq = torch.where(p1_rew != p_p1_rew)[0].cpu().numpy()
            result += f'    Found different rewards for player 1 at env indices: {neq}'
            is_same = False
        if (p2_rew != p_p2_rew).any():
            neq = torch.where(p2_rew != p_p2_rew)[0].cpu().numpy()
            result += f'    Found different rewards for player 2 at env indices: {neq}'
            is_same = False
        if (p1_s_rew != p_p1_s_rew).any():
            neq = torch.where(p1_s_rew != p_p1_s_rew)[0].cpu().numpy()
            result += f'    Found different shaped rewards for player 1 at env indices: {neq}'
            is_same = False
        if (p2_s_rew != p_p2_s_rew).any():
            neq = torch.where(p2_s_rew != p_p2_s_rew)[0].cpu().numpy()
            result += f'    Found different shaped rewards for player 2 at env indices: {neq}'
            is_same = False
        if not is_same:
            print(f' Layout {layout_name}: FAIL.\n{result}')
            return False
        total_reward += p1_rew.sum().item() + p2_rew.sum().item()
        total_shaped_reward += p1_s_rew.sum().item() + p2_s_rew.sum().item()
    
    print(f'Layout {layout_name}: PASS | Total reward = {total_reward} | Total shaped reward = {total_shaped_reward}')
    return True

def test_layout(layout_name: str = SIMPLE_O, seed: int = 137, batch_size: int = 256, horizon: int = 400):
    """
    合并测试：同时验证状态转移和奖励（稀疏 + shaping）是否正确。
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    actions = torch.randint(0, len(all_actions), (horizon, batch_size, 2), generator=generator, device=device)

    # 参考环境（逐个）
    envs = [OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(layout_name), horizon=horizon) for _ in range(batch_size)]
    # 并行环境，开启 reward shaping 以便同时测试 shaped reward
    parallel_env = ParallelEnvironment(layout_name, batch_size, horizon, enable_reward_shaping=True)

    total_sparse_reward = 0
    total_shaped_reward = 0

    for step_id, step_actions in enumerate(tqdm(actions, desc=f'{layout_name}')):
        states: List[OvercookedState] = []
        sparse_rewards = []      # 每个环境返回的 [p1_sparse, p2_sparse]
        shaped_rewards = []      # 每个环境返回的 [p1_shaped, p2_shaped]

        # 1. 执行所有参考环境
        for env_id, (env, action) in enumerate(zip(envs, step_actions)):
            p1_action, p2_action = action.tolist()
            joint_action = (all_actions[p1_action], all_actions[p2_action])
            state, _, _, info = env.step(joint_action)
            states.append(state)
            sparse_rewards.append(info["sparse_r_by_agent"])
            shaped_rewards.append(info["shaped_r_by_agent"])

        # 2. 执行并行环境
        done, p_p1_rew, p_p2_rew, p_p1_s_rew, p_p2_s_rew, parallel_state = parallel_env.step(
            step_actions[:, 0], step_actions[:, 1]
        )

        # 3. 验证状态是否一致
        state_result = compare_states(states, parallel_state, step_id)
        if state_result is not None:
            print(f'Layout {layout_name}: FAIL (state mismatch)')
            print(state_result)
            return False

        # 4. 验证奖励是否一致
        sparse_tensor = torch.tensor(sparse_rewards, dtype=torch.long, device=device)
        shaped_tensor = torch.tensor(shaped_rewards, dtype=torch.long, device=device)
        p1_sparse = sparse_tensor[:, 0]
        p2_sparse = sparse_tensor[:, 1]
        p1_shaped = shaped_tensor[:, 0]
        p2_shaped = shaped_tensor[:, 1]

        is_same = True
        error_msg = f' Error at step {step_id}:'

        if (p1_sparse != p_p1_rew).any():
            neq = torch.where(p1_sparse != p_p1_rew)[0].cpu().numpy()
            error_msg += f'\n  Player 1 sparse reward mismatch at env indices: {neq}'
            is_same = False
        if (p2_sparse != p_p2_rew).any():
            neq = torch.where(p2_sparse != p_p2_rew)[0].cpu().numpy()
            error_msg += f'\n  Player 2 sparse reward mismatch at env indices: {neq}'
            is_same = False
        if (p1_shaped != p_p1_s_rew).any():          # 修复：原代码误用了 p1_sparse
            neq = torch.where(p1_shaped != p_p1_s_rew)[0].cpu().numpy()
            error_msg += f'\n  Player 1 shaped reward mismatch at env indices: {neq}'
            is_same = False
        if (p2_shaped != p_p2_s_rew).any():          # 同理修复
            neq = torch.where(p2_shaped != p_p2_s_rew)[0].cpu().numpy()
            error_msg += f'\n  Player 2 shaped reward mismatch at env indices: {neq}'
            is_same = False

        if not is_same:
            print(f'Layout {layout_name}: FAIL (reward mismatch)')
            print(error_msg)
            return False

        # 5. 累计总奖励（可选）
        total_sparse_reward += p1_sparse.sum().item() + p2_sparse.sum().item()
        total_shaped_reward += p1_shaped.sum().item() + p2_shaped.sum().item()

    print(f'Layout {layout_name}: PASS | Total sparse reward = {total_sparse_reward} | Total shaped reward = {total_shaped_reward}')
    return True


test_layout('limited_0', seed=40000)
exit()

for layout_name in all_available_layouts:
    same = test_layout(layout_name)
    if not same:
        exit()

