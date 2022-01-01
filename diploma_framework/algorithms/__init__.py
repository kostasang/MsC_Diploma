from ._ppo import PPO
from ._deep_q_learning import DQN
from ._reinforce import Reinforce
from ._a3c import A3C

import logging

logger = logging.getLogger("deepRL")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)