from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

# celui-ci est pour l'apprentissage avec la fonction reset()

class empty_agent1(base_agent.BaseAgent):
  # one-time setup
  def __init__(self):
    super(empty_agent1, self).__init__()
    
  # before each game
  def reset(self):
    super(empty_agent1, self).reset()

  # read state from obs and ALWAYS act (it's necessary)
  def step(self, obs):
    super(empty_agent1, self).step(obs)

    return actions.RAW_FUNCTIONS.no_op()

# sinon la il y a encore plus simple

class empty_agent2(base_agent.BaseAgent):
  def step(self, obs):
    super(empty_agent, self).step(obs)
    return actions.RAW_FUNCTIONS.no_op()
    
