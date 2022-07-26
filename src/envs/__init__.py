from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
from multiagent_mujoco.mujoco_multi import MujocoMulti
import sys
import os


def env_fn(env, **kwargs):
    return env(**kwargs)


REGISTRY = {
    "mujoco_multi": partial(env_fn, env=MujocoMulti)

}

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
