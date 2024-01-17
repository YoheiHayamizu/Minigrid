from minigrid.envs import LockedRoomEnv, RedBlueDoorEnv
# from minigrid.multi_envs import LockedRoomEnv as LockedRoomEnvMultiGrid
# import matplotlib; matplotlib.use('TkAgg')
import time
import numpy as np


def random_walk(num_episodes=1, render=False):
    """
    Visualize a trajectory where agents are taking random actions.
    """
    kwargs = {
        # 'see_through_walls': True,
    }
    if render:
        kwargs.update({'render_mode': 'human', 'screen_size': 500})

    # env = LockedRoomEnv(**kwargs)
    env = RedBlueDoorEnv(**kwargs)
    # env = LockedRoomEnv(render_mode='human', screen_size=500)

    for episode in range(num_episodes):
        print('Episode', episode)
        obs, _ = env.reset(seed=episode)
        terminated, truncated = False, False
        while not (terminated or truncated):
            if render:
                env.render()
            random_action = env.np_random.integers(env.action_space.n)
            obs, reward, terminated, truncated, _ = env.step(random_action)
        # print('Episode:', episode, 'Score:', env.score)


if __name__ == '__main__':
    random_walk(1, render=True)
    import cProfile
    cProfile.run('random_walk(1000)', sort='cumtime')
