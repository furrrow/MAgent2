# from magent2.environments import battle_v4, battlefield_v5
from magent2.environments.custom_map import battlefield
from pettingzoo.utils import random_demo

# env = battle_v4.env(render_mode='human')
env = battlefield.env(render_mode='human')
random_demo(env, render=True, episodes=1)