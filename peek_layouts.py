from env.constants.layouts import LAYOUT_DIR, all_layouts
from typing import List, Tuple, Dict, Optional, Set

import os

all_keys: Set[str] = set([
    'grid',
    'start_state',
    'start_all_orders',
    'start_bonus_orders',
    'order_bonus',
    'recipe_times',
    'recipe_values',
    'onion_time',
    'tomato_time',
    'onion_value',
    'tomato_value',
    'rew_shaping_params',
])



for layout in all_layouts:
    layout_path = os.path.join(LAYOUT_DIR, f'{layout}.layout')
    with open(layout_path, 'r', encoding='utf-8') as f:
        layout_dict = eval(f.read())
    
    # if 'start_state' in layout_dict:
    #     print(layout)

    # if 'order_bonus' in layout_dict:
    #     print(layout)

    

    # grid: List[str] = [row.strip() for row in layout_dict['grid'].split('\n')]
    # for row in grid:
    #     print(row)
    # keys = layout_dict.keys()
    # all_keys = all_keys & set(keys)
    # print(layout_dict)
    # break

# for key in all_keys:
#     print(f'    \'{key}\',')


'''
D:/Anaconda/ANACONDA/envs/MultiAgent/Lib/site-packages/overcooked_ai_py/data/layouts/bonus_order_test.layout
'''
