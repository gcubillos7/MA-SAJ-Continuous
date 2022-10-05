REGISTRY = {}

from .masaj_episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .masaj_parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
