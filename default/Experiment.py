# Built to run RLGlue experiment
# Noah Burghardt

import numpy as np
from Agent import Agent
from Environment import Environment
from rl_glue import RLGlue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
	
	agent = Agent()
	env = Environment()
	rlglue = RLGlue(env, agent)
	del agent, env

if __name__ == '__main__':
	main()
