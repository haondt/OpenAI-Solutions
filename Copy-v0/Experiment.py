# Built to run RLGlue experiment
# Noah Burghardt

import numpy as np
from Agent import Agent
from Environment import Environment
from rl_glue import RLGlue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random

class testAgent:
	policy = None
	def __init__(self, policy):
		self.policy = policy
	def agent_init(self):
		pass
	def agent_start(self, state):
		return self.policy[state]
	def agent_step(self, reward, state):
		return self.policy[state]
	def agent_end(self, reward):
		pass
	def agent_message(self, message):
		pass

# Measure the agents policy by its ability to make a perfect copy
# of the tape (binary value) and average over 100 episodes.

def testPolicy(policy):
	env = Environment()
	agent = testAgent(policy)
	rlglue = RLGlue(env, agent)
	rlglue.rl_init()
	#rlglue.rl_env_message('renderON')
	performance = 0
	for ep in range(100):
		rlglue.rl_start()
		terminal = False
		reward = None
		while not terminal:
			reward, state, action, terminal = rlglue.rl_step()
		if reward > 0:
			performance += 1


	return performance / 100

def main():
	
	num_eps = 50000
	num_runs = 10
	random.seed(0)
	np.random.seed(0)
	agent = Agent()
	env = Environment()
	rlglue = RLGlue(env, agent)
	del agent, env
	for run in range(num_runs):
		rlglue.rl_init()
		performances = []
		for ep in range(num_eps):
			rlglue.rl_start()
			#rlglue.rl_env_message('renderON')
			terminal = False
			while not terminal:
				reward, state, action, terminal = rlglue.rl_step()
			
			
			# Find the first policy that performs at 100%
			performance = testPolicy(rlglue.rl_agent_message('policy')) * 100
			performances.append(performance)
			if performance >= 100:
				#print(rlglue.rl_agent_message('policy'))
				print('Episode: %d' % (ep+1))
				break
		plt.plot(performances)
	plt.savefig('test.png')

			
if __name__ == '__main__':
	main()
