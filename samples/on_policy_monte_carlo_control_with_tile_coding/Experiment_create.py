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
	
	# set up 2d array for average rewards
	# rewards[step] = sum of rewards across all runs for that step
	rewards = [0 for i in range(1000)]
	for run in range(1):
		rlglue.rl_init()
		#rlglue.rl_env_message('renderON')
		rlglue.rl_start()
		
		terminal = False
		for step in range(1000):
			if not terminal:
				r, s, a, terminal = rlglue.rl_step()
				rewards[step] += r

	# average rewards
	rewards = [i/1 for i in rewards]

	return rewards


def main():
	# Seed rng's for consistent testing
	random.seed(0)
	np.random.seed(0)

	# Generate agent, environment and RLGlue
	env = Environment()
	env.env_init()
	agent = Agent(env.get_actions(),
		env.get_max_observation(),
		env.get_min_observation())
	rlglue = RLGlue(env, agent)
	del agent, env


	# Configure experiment
	num_eps = 100000
	# initialize rlglue
	rlglue.rl_init()

	avg_rewards = []
	avg_reward = 0
	max_reward = 0
	best_policy = None
	# Run through each episode
	#rlglue.rl_env_message('renderON')
	#for ep in range(num_eps):
	ep = 0
	x = 50
	last_i = x
	last_n = np.zeros(x)
	best = (0,-1)
	while ep < num_eps:
		last_i += 1
		if last_i >= len(last_n):
			last_i = 0
		ep += 1
		#if ep % int(num_eps/10) == 0:
		#print('ep:', ep, 'bestpolicy', max_reward)
		# start episode
		rlglue.rl_start()
		rewards = 0
		steps = 1
		# Run episode to its completion
		terminal = False
		while not terminal:
			reward, state, action, terminal = rlglue.rl_step()
			rewards += reward
			steps += 1

		if steps > best[0]:
			best = (steps,ep)

		#avg_reward = rewards
		#avg_rewards.append(avg_reward)
		last_n[last_i] = steps

		#print('ep',ep, 'steps', steps)
		#print('ep:',ep, 'avg reward:', avg_reward, 'steps:', steps)
		#print(rlglue.rl_agent_message('policy'))
		#input()
		#if best[0] >= 500:
		if np.average(last_n) > 200:
			print('ep',ep,'mvg avg', np.average(last_n), 'best',best)
			break
	
		
	plt.plot(avg_rewards)
	plt.plot(moving_average(avg_rewards,10))
	plt.plot(moving_average(avg_rewards,100))
	plt.savefig('results.png')


def moving_average(a, n=3):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n-1:]/n
	

if __name__ == '__main__':
	main()
