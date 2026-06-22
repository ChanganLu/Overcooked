import torch
from torch import LongTensor
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, ObjectState, PlayerState, SoupState
from typing import Tuple, List, Dict, Optional

from env.constants.items import ITEM_EMPTY, ITEM_ONION, ITEM_TOMATO, ITEM_DISH, ITEM_SOUP

def extract_state_info(state: OvercookedState, H: int, W: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int, int], LongTensor]:
    """
    从 OvercookedState 中提取当前状态，返回元组：
    (
        player1_pos, player1_ori, player1_item,
        player2_pos, player2_ori, player2_item,
        object_tensor
    )
    其中：
        player_pos : (x, y)
        player_ori  : (dx, dy)
        player_item : (id, num_onion, num_tomato)
        object_tensor : pytorch tensor (4, H, W)
    """

    # 提取玩家信息（固定2个玩家）
    players: List[PlayerState] = state.players
    assert len(players) == 2
    player_infos = []
    for p in players:
        pos: Tuple[int, int] = p.position          # (x, y)
        ori: Tuple[int, int] = p.orientation       # (dx, dy)
        held: Optional[ObjectState] = p.held_object

        if held is None:
            item = (ITEM_EMPTY, 0, 0)
        else:
            held: ObjectState
            if held.name == 'soup':
                held: SoupState
                ingredients = held.ingredients
                onion = ingredients.count('onion')
                tomato = ingredients.count('tomato')
                item = (ITEM_SOUP, onion, tomato)
            else:
                if held.name == 'onion':
                    tid = ITEM_ONION
                elif held.name == 'tomato':
                    tid = ITEM_TOMATO
                elif held.name == 'dish':
                    tid = ITEM_DISH
                else:
                    tid = ITEM_EMPTY
                item = (tid, 0, 0)
        player_infos.append((pos, ori, item))

    # 按玩家顺序输出
    p1_pos, p1_ori, p1_item = player_infos[0]
    p2_pos, p2_ori, p2_item = player_infos[1]

    # 初始化其他物品张量 (4, H, W)，默认 0, 0, 0, -1
    obj_tensor = torch.zeros((4, H, W), dtype=torch.int32)
    obj_tensor[-1] = -1

    # 填充非玩家持有的物品（state.objects）
    object_list: List[Tuple[Tuple[int, int], ObjectState]] = list(state.objects.items())
    for pos, obj in object_list:
        x, y = pos
        if obj.name == 'onion':
            tid, onion, tomato, countdown = ITEM_ONION, 0, 0, -1
        elif obj.name == 'tomato':
            tid, onion, tomato, countdown = ITEM_TOMATO, 0, 0, -1
        elif obj.name == 'dish':
            tid, onion, tomato, countdown = ITEM_DISH, 0, 0, -1
        elif obj.name == 'soup':
            tid = ITEM_SOUP
            obj: SoupState
            # 获取配方
            ingredients = obj.ingredients  # list of strings
            onion = ingredients.count('onion')
            tomato = ingredients.count('tomato')
            # 烹饪状态
            if obj._cooking_tick == -1:          # 尚未开始烹饪
                countdown = -1
            elif obj.is_ready:                   # 已烹饪完成
                countdown = 0
            else:                                # 正在烹饪，剩余时间
                countdown = obj.cook_time - obj._cooking_tick
        else:
            tid, onion, tomato, countdown = ITEM_EMPTY, 0, 0, -1

        # 张量索引: (channel, y, x)
        obj_tensor[0, y, x] = tid
        obj_tensor[1, y, x] = onion
        obj_tensor[2, y, x] = tomato
        obj_tensor[3, y, x] = countdown

    return p1_pos, p1_ori, p1_item, p2_pos, p2_ori, p2_item, obj_tensor
