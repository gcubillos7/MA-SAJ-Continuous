REGISTRY = {}

from .rnn_agent import RNNAgent
from .rode_agent import RODEAgent
from .mlp_agent import MLPAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rode"] = RODEAgent
REGISTRY["mlp"] = MLPAgent