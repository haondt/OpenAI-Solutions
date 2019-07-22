# Agent
# Based off the agent define in the RLGlue Software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0

# Modified by Noah Burghardt
import numpy as np

# Agent class to represent a particular RL strategy
class Agent:

	Q = None
	returns = None
	pi = None

	# Variables to hold the execution of an episode
	ep_states = None
	ep_actions = None
	ep_rewards = None

	# Action space
	A = None

	# Agent configuration variables
	epsilon = 0.1
	gamma = 0.9
	
	# Declare agent variables
	def __init__(self, actions):
		self.Q = {}
		self.returns = {}
		self.A = actions
		self.pi = {}

	# Initialize agent variables
	# Run once, in experiments
	def agent_init(self):
		# Assume there are no states (new states can be valued lazily)
		self.Q = {}
		self.pi = {}
		self.returns = {}

	# Start agent
	# Runs at the beginning of an episode. The first method called when the experiment
	# starts, called after the environment starts
	# Args:
	#	state (state observation): The agent's current state
	# returns:
	#	The first action the agent takes
	def agent_start(self, state):
		# start episode
		a = self.epGreedy(self.epsilon, self.pi, state, self.A)

		# Reset episode data and store new
		self.ep_states = [state]
		self.ep_actions = [a]
		self.ep_rewards = [None]


		return a

	# A step taken by the agent
	# Args:
	#	reward (float): the reward received for taking the last action taken
	#	state (state observation): The agen's current state
	# returns:
	#	The action the agent is taking
	def agent_step(self, reward, state):
		# Choose ep_greedy action
		a = self.epGreedy(self.epsilon, self.pi, state, self.A)

		# Save episode step
		self.ep_rewards.append(reward)
		self.ep_states.append(state)
		self.ep_actions.append(a)


		return a
	
	# Run when the agent termintates
	# Args:
	#	reward (float): the reward the agent received for entering the terminal state
	def agent_end(self, reward):
		# finish off episode
		self.ep_rewards.append(reward)

		# Append terminal state
		self.ep_states.append(None)

		# set up variables for processing episode

		# Loop through steps in episode
		G = {}
		Return = 0

		for t in range(len(self.ep_actions)):
			# get reward, state-action pair and reward for episode step
			r = self.ep_rewards[t+1]
			Return 	
# TODO: calculat return (gotta do this biz in reverse):
# G_t = R_{t+1} + gamma*G_{t+1}
			sap = (self.ep_states[t], self.ep_actions[t])

			if sap not in G:
				G[sap] = # return 
				self.returns[sap] = []
			self.returns[sap].append(G[sap])



			if sap[0] == (6,4):
				print(sap, self.Q.get(sap,None))
			self.Q[sap] = sum(self.returns[sap])/len(self.returns[sap])
			if sap[0] == (6,4):
				print(sap, self.Q[sap], self.returns[sap])
				input()

			Astar = self.argmaxa_rand(self.Q, sap[0],self.A)
			for a in self.A:
				if a == Astar:
					self.pi[(sap[0],a)] = 1 - self.epsilon + self.epsilon / len(self.A)
				else:
					self.pi[(sap[0],a)] = self.epsilon / len(self.A)
			


	# Receive a message form RLGlue
	# Args:
	#	Message (str): the message passed
	# returns:
	#	response (str): The agent's response to the message (optional)
	def agent_message(self, message):
		if message == "policy":
			return self.pi

	# Perform an action selection using an epsilon-greedy selection
	# Args
	#	eps: epsilon value
	#	Q: action values to base selection off of
	#	A: actions to choose from
	#	s: current state
	# returns:
	#	action (action): action chosen
	def epGreedy(self, eps, Q, s, A):
		# Best action with probability 1-eps
		if np.random.random() > eps:
			return self.argmaxa_rand(Q, s, A)

		# Choose an action at random with porbability eps
		else:
			return np.random.choice(A)
	


	# returns the action with the highest action-value
	# Args:
	#	Q: Action values
	#	s: state to evaluate
	#	A: Actions to pick from
	def argmaxa(self, Q, s, A):
		
		# Find the highest action value
		maxQ = max([Q.get((s,a),0) for a in A])
		# Find actions with highest value
		best_actions = [a for a in A if Q.get((s,a),0) == maxQ]
		# consistently pick from best actions
		return sorted(best_actions)[0]

			
	# returns the action with the highest action-value
	# Args:
	#	Q: Action values
	#	s: state to evaluate
	#	A: Actions to pick from
	def argmaxa_rand(self, Q, s, A):
		# Find the highest action value
		maxQ = max([Q.get((s,a),0) for a in A])
		# Find actions with highest value
		best_actions = [a for a in A if Q.get((s,a),0) == maxQ]
		# Randomly pick from actions with top action value
		return np.random.choice(best_actions)
