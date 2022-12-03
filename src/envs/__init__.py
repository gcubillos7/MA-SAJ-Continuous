from functools import partial
import sys
import os

try:
    from .mamujoco import ManyAgentAntEnv, ManyAgentSwimmerEnv, MujocoMulti
    # from multiagent_mujoco.mujoco_multi import MujocoMulti
    def env_fn(env, **kwargs):
        return env(**kwargs)
    REGISTRY = {
        "mujoco_multi": partial(env_fn, env=MujocoMulti)
    }

except:
    try:
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
    except:
        from .multiagentenv import MultiAgentEnv
        # from .matrix_game.cts_matrix_game import Matrixgame as CtsMatrix
        # from .particle import Particle
        from .mamujoco import ManyAgentAntEnv, ManyAgentSwimmerEnv, MujocoMulti
        def env_fn(env, **kwargs) -> MultiAgentEnv:
            # env_args = kwargs.get("env_args", {})
            return env(**kwargs)


        REGISTRY = {}
        # REGISTRY["cts_matrix_game"] = partial(env_fn, env=CtsMatrix)
        # REGISTRY["particle"] = partial(env_fn, env=Particle)
        REGISTRY["mujoco_multi"] = partial(env_fn, env=MujocoMulti)
        REGISTRY["manyagent_swimmer"] = partial(env_fn, env=ManyAgentSwimmerEnv)
        REGISTRY["manyagent_ant"] = partial(env_fn, env=ManyAgentAntEnv)
