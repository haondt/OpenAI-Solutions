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
	
	num_eps = 2000

	agent = Agent()
	env = Environment()
	rlglue = RLGlue(env, agent)
	del agent, env
	solves = 0
	rlglue.rl_init()
	rewards = []
	for ep in range(num_eps):
		rlglue.rl_start()
		#rlglue.rl_env_message('renderON')
		terminal = False
		reward = 0
		while not terminal:
			reward, state, action, terminal = rlglue.rl_step()
		rewards.append(reward)
		if ep >= 99:
			if np.average(rewards[ep-99:ep+1]) > 0.78:
				print('solved at episode %d' % ep+1)

if __name__ == '__main__':
	main()
