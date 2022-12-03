from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .masaj_learner import MASAJ_Learner
from .masaj_learner_simple import MASAJ_Simple


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["masaj_learner"] = MASAJ_Learner
REGISTRY["masaj_simple_learner"] = MASAJ_Simple
