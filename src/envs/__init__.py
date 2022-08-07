from functools import partial
import sys
import os

#
try:
    from multiagent_mujoco.mujoco_multi import MujocoMulti

    def env_fn(env, **kwargs):
        return env(**kwargs)


    REGISTRY = {
        "mujoco_multi": partial(env_fn, env=MujocoMulti)

    }


except:

    from smac.env import MultiAgentEnv, StarCraft2Env

    discrete = True


    def env_fn(env, **kwargs) -> MultiAgentEnv:
        return env(**kwargs)


    REGISTRY = {
        "sc2": partial(env_fn, env=StarCraft2Env)

    }
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
