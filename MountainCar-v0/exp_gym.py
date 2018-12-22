#!/usr/bin/env python

import numpy as np
from tqdm import tqdm
from agent_gym import Agent

from rl_glue import RLGlue
from env_gym import Environment


def question_1():
# Specify hyper-parameters

	agent = Agent()
	environment = Environment()
	rlglue = RLGlue(environment, agent)
	np.random.seed(0)
	num_episodes = 200
	see_eps = [157]
	num_runs = 1
	max_eps_steps = 100000

	# test with various stepsizes (alphas) for agent
	stepSizes = np.linspace(0.01,1,100)
	# best stepsize so far (comment out to test many)
	stepSizes = [0.559184]

	# seperate run for each stepsize
	for step in stepSizes:
		
		# initialize agent and software, with chosen stepsize
		rlglue.rl_init()
		rlglue.rl_agent_message('step:' + str(step))

		# keep track of total rewards for each episode
		total_rewards = []

		for ep in range(num_episodes):
			# render only selected episodes
			if ep in see_eps:
				rlglue.rl_env_message('rOFF')
			if ep+1 in see_eps:
				rlglue.rl_env_message('rON')
				print("Episode %d" % (ep+1))
			
			# initializse for episode
			rlglue.rl_start()
			terminal = False
			total_reward = 0

			# run episode and calculate total reward
			while not terminal:
				reward, state, action, terminal = rlglue.rl_step()
				total_reward += reward
			total_rewards.append(total_reward)
			
			# calculate average reward of the last 100 episodes
			if ep >= 99:
				total = np.sum(total_rewards[ep-99:ep+1])
				avg = total/100

				# check if results indicate the problem is solved
				if avg > -110:
					print("Solved at episode %d, avg reward: %f" % (ep+1, avg))
					break
	

	# close environment 
	environment.close()
if __name__ == "__main__":
	question_1()
