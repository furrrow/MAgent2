# noqa
"""
## Battlefield

```{figure} battlefield.gif
:width: 140px
:name: battlefield
```

| Import             | `from magent2.environments import battlefield_v4` |
|--------------------|------------------------------------------------|
| Actions            | Discrete                                       |
| Parallel API       | Yes                                            |
| Manual Control     | No                                             |
| Agents             | `agents= [red_[0-11], blue_[0-11]]`            |
| Agents             | 24                                             |
| Action Shape       | (21)                                           |
| Action Values      | Discrete(21)                                   |
| Observation Shape  | (13,13,5)                                      |
| Observation Values | [0,2]                                          |
| State Shape        | (80, 80, 5)                                    |
| State Values       | (0, 2)                                         |


Same as [battle](./battle) but with fewer agents arrayed in a larger space with obstacles.

A small-scale team battle, where agents have to figure out the optimal way to coordinate their small team in a large space and maneuver around obstacles in order to defeat the opposing team. Agents are rewarded for their individual performance, and not for the performance of their neighbors, so
coordination is difficult.  Agents slowly regain HP over time, so it is best to kill an opposing agent quickly. Specifically, agents have 10 HP, are damaged 2 HP by each attack, and recover 0.1 HP every turn.

Like all MAgent2 environments, agents can either move or attack each turn. An attack against another agent on their own team will not be registered.

### Arguments

``` python
battlefield_v5.env(map_size=80, minimap_mode=False, step_reward=-0.005,
dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
max_cycles=1000, extra_features=False)
```

`map_size`: Sets dimensions of the (square) map. Minimum size is 46.

`minimap_mode`: Turns on global minimap observations. These observations include your and your opponents piece densities binned over the 2d grid of the observation space. Also includes your `agent_position`, the absolute position on the map (rescaled from 0 to 1).

`step_reward`:  reward added unconditionally

`dead_penalty`:  reward added when killed

`attack_penalty`:  reward added for attacking

`attack_opponent_reward`:  Reward added for attacking an opponent

`max_cycles`:  number of frames (a step for each agent) until game terminates

`extra_features`: Adds additional features to observation (see table). Default False

#### Action Space

Key: `move_N` means N separate actions, one to move to each of the N nearest squares on the grid.

Action options: `[do_nothing, move_12, attack_8]`

#### Reward

Reward is given as:

* 5 reward for killing an opponent
* -0.005 reward every step (step_reward option)
* -0.1 reward for attacking (attack_penalty option)
* 0.2 reward for attacking an opponent (attack_opponent_reward option)
* -0.1 reward for dying (dead_penalty option)

If multiple options apply, rewards are added.

#### Observation space

The observation space is a 13x13 map with the below channels (in order):

feature | number of channels
--- | ---
obstacle/off the map| 1
my_team_presence| 1
my_team_hp| 1
my_team_minimap(minimap_mode=True)| 1
other_team_presence| 1
other_team_hp| 1
other_team_minimap(minimap_mode=True)| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)| 21
last_reward(extra_features=True)| 1
agent_position(minimap_mode=True)| 2

### State space

The observation space is a 80x80 map. It contains the following channels, which are (in order):

feature | number of channels
--- | ---
obstacle map| 1
team_0_presence| 1
team_0_hp| 1
team_1_presence| 1
team_1_hp| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)|  21
last_reward(extra_features=True)| 1



### Version History

* v0: Initial MAgent2 release (0.3.0)

"""

import math

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

import magent2
from magent2.environments.battle.battle import KILL_REWARD, get_config
from magent2.environments.magent_env import magent_parallel_env, make_env


default_map_size = 80
max_cycles_default = 1000
minimap_mode_default = False
default_reward_args = dict(
    step_reward=-0.005,
    dead_penalty=-0.1,
    attack_penalty=-0.1,
    attack_opponent_reward=0.2,
)


def parallel_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    render_mode=None,
    seed=None,
    **reward_args,
):
    env_reward_args = dict(**default_reward_args)
    env_reward_args.update(reward_args)
    return _parallel_env(
        map_size,
        minimap_mode,
        env_reward_args,
        max_cycles,
        extra_features,
        render_mode,
        seed,
    )


def raw_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    seed=None,
    **reward_args,
):
    return parallel_to_aec_wrapper(
        parallel_env(
            map_size, max_cycles, minimap_mode, extra_features, seed=seed, **reward_args
        )
    )


env = make_env(raw_env)


class _parallel_env(magent_parallel_env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "battlefield_v5",
        "render_fps": 5,
    }

    def __init__(
        self,
        map_size,
        minimap_mode,
        reward_args,
        max_cycles,
        extra_features,
        render_mode=None,
        seed=None,
    ):
        EzPickle.__init__(
            self,
            map_size,
            minimap_mode,
            reward_args,
            max_cycles,
            extra_features,
            render_mode,
            seed,
        )
        assert map_size >= 46, "size of map must be at least 46"
        env = magent2.GridWorld(
            get_config(map_size, minimap_mode, seed, **reward_args), map_size=map_size
        )
        self.leftID = 0
        self.rightID = 1
        reward_vals = np.array([KILL_REWARD] + list(reward_args.values()))
        reward_range = [
            np.minimum(reward_vals, 0).sum(),
            np.maximum(reward_vals, 0).sum(),
        ]
        names = ["red", "blue"]
        super().__init__(
            env,
            env.get_handles(),
            names,
            map_size,
            max_cycles,
            reward_range,
            minimap_mode,
            extra_features,
            render_mode,
        )

    def get_block_obstacles(self, x0, x1, y0, y1):
        top = [[x, y0] for x in range(x0, x1)]
        bot = [[x, y1] for x in range(x0, x1)]
        left = [[x0, y] for y in range(y0, y1)]
        right = [[x1, y] for y in range(y0, y1+1)]
        return top + bot + left + right

    def generate_map(self):
        env, map_size, handles = self.env, self.map_size, self.handles
        """ generate a map, which consists of two squares of agents"""
        # width = height = map_size
        # init_num = map_size * map_size * 0.04

        width = map_size # 80
        height = int(map_size * 3/4)  # 60

        l_gap = 20
        r_gap = 1
        leftID, rightID = 0, 1

        # left
        pos = []
        pos += self.get_block_obstacles(10, 15, 10, 15)
        pos += self.get_block_obstacles(10, 15, 45, 50)
        pos += self.get_block_obstacles(25, 30, 25, 45)
        pos += self.get_block_obstacles(50, 55, 15, 45)
        for y in range(1, 45):
            pos.append((width / 2 - 5, y))
            pos.append((width / 2 - 4, y))
        for y in range(50, height-1):
            pos.append((width / 2 - 5, y))
            pos.append((width / 2 - 4, y))

        for y in range(1 , 27):  # 1 -> 30
            pos.append((width / 2 + 7, y))
            pos.append((width / 2 + 6, y))

        for y in range(32                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      , height - 1): # 35 -> 59
            pos.append((width / 2 + 7, y))
            pos.append((width / 2 + 6, y))

        for x, y in pos:
            if not (0 < x < width - 1 and 0 < y < height - 1):
                assert False
        env.add_walls(pos=pos, method="custom")

        n_l = 16
        side = int(math.sqrt(n_l)) * 2
        pos = []
        for x in range(width // 2 - l_gap - side, width // 2 - l_gap - side + side, 2):
            for y in range((height - side) // 2, (height - side) // 2 + side, 2):
                pos.append([x, y, 0])

        for x, y, _ in pos:
            if not (0 < x < width - 1 and 0 < y < height - 1):
                assert False
        env.add_agents(handles[leftID], method="custom", pos=pos)

        # right
        n_r = 4
        side = int(math.sqrt(n_r)) * 2
        pos = []
        for x in range(width // 2 + r_gap, width // 2 + r_gap + side, 2):
            for y in range((height - side) // 2, (height - side) // 2 + side, 2):
                pos.append([x, y, 0])

        for x, y, _ in pos:
            if not (0 < x < width - 1 and 0 < y < height - 1):
                assert False
        env.add_agents(handles[rightID], method="custom", pos=pos)
