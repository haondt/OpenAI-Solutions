# Built to run RLGlue experiment
# Noah Burghardt

import numpy as np
import pickle
from Agent import Agent
#from WindyGrid import GridWorld as Environment
from Environment import Environment
from rl_glue import RLGlue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# Define an agent that acts entirely based on the input policy
class testAgent:
	policy = None
	def __init__(self, policy):
		self.policy = policy
	def agent_init(self):
		pass
	def agent_start(self, state):
		return self.apply_policy(state)
	def agent_step(self, reward, state):
		return self.apply_policy(state)
	def agent_end(self, reward):
		pass
	def agent_message(self, message):
		pass
	def apply_policy(self, state):
		if state not in self.policy:
			return random.choice(list(self.policy.values()))
		return self.policy[state]


# Measure the given policy
# Spits out a 2d array of the average reward per step over the first 1000 steps of an episode.
# Averaged across 100 runs.
def testPolicy(policy):
	agent = testAgent(policy)
	env = Environment()
	rlglue = RLGlue(env, agent)
	del env, agent
	rlglue.rl_init()
	
	for run in range(1):
		rlglue.rl_init()
		rlglue.rl_env_message('renderON')
		rlglue.rl_start()

		total_reward = 0
		terminal = False
		while not terminal:
			r, s, a, terminal = rlglue.rl_step()
			total_reward += r


	return total_reward


def main():
	# Seed rng's for consistent testing
	random.seed(0)
	np.random.seed(0)

	# Generate agent, environment and RLGlue
	env = Environment()
	agent = Agent(env.get_actions())
	rlglue = RLGlue(env, agent)
	del agent, env


	# Get generated policy
	policy = pickle.load(open('policy.pickle','rb'))

	# Test policy
	result = testPolicy(policy)
	print('result:', result)

	# Graph results
	#plt.plot(result)
	#plt.savefig('results.png')

def moving_average(a, n=3):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n-1:]/n
	

if __name__ == '__main__':
	main()
