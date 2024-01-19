from minigrid.envs import LockedRoomEnv, RedBlueDoorEnv
from minigrid.core.actions import Actions
from minigrid.envs import LockedRoomEnv
# import matplotlib; matplotlib.use('TkAgg')
import time
import numpy as np
import io
import pstats
import cProfile


num_actions = len(Actions)


def item(x):
    if isinstance(x, dict):
        return x.items()


def prof_to_csv(prof: cProfile.Profile, sort=-1):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).print_stats(sort)
    result = out_stream.getvalue()
    # chop off header lines
    result = 'ncalls' + result.split('ncalls')[-1]
    lines = [','.join(line.rstrip().split(None, 5)) for line in result.split('\n')]
    return '\n'.join(lines)


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
    env = RedBlueDoorEnv(**kwargs, agents=5)
    # env = LockedRoomEnv(**kwargs, agents=5)

    for episode in range(num_episodes):
        print('Episode', episode)
        obs, _ = env.reset(seed=episode)
        terminated = {agent_id: False for agent_id in env.agents}
        truncated = {agent_id: False for agent_id in env.agents}

        while not all(truncated.values()) and not all(terminated.values()):
            if render:
                env.render()

            random_action = {
                agent_id: env.np_random.integers(num_actions)
                for agent_id in env.agents
            }
            # print(f"Action: {[a for a in list(env.actions)]}")
            # random_action[0] = int(input())
            obs, reward, terminated, truncated, _ = env.step(random_action)


if __name__ == '__main__':
    random_walk(1, render=True)
    pr = cProfile.Profile()
    pr.run('random_walk(100)')
    csv = prof_to_csv(pr)
    with open('profile.csv', 'w') as f:
        f.write(csv)
