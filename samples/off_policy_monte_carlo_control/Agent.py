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
		a = self.pi_select(self.pi, state,self.A)

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
		a = self.pi_select(self.pi,state, self.A)

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

		T = len(self.ep_actions)

		# Get the returns for each step in episode
		Return = 0
		G = {}
		for t in range(T-1, -1, -1):
			r = self.ep_rewards[t+1]
			sap = (self.ep_states[t], self.ep_actions[t])
			Return = r + self.gamma*Return
			G[sap] = Return

		for t in range(T):
			# get reward, state-action pair and reward for episode step
			sap = (self.ep_states[t], self.ep_actions[t])

			#print('t', t, 'sap', sap,'G', Return)

			self.returns[sap] = self.returns.get(sap,[]) + [G[sap]]


			#input()

			#if sap[0] == (6,4):
				#print(sap, self.Q.get(sap,None))
			self.Q[sap] = sum(self.returns[sap])/len(self.returns[sap])
			#if sap[0] == (6,4):
				#print(sap, self.Q[sap], self.returns[sap])
				#input()

			Astar = self.argmaxa_rand(self.Q, sap[0],self.A)
			self.pi[sap[0]] = {}
			for a in self.A:
				if a == Astar:
					self.pi[sap[0]][a] = 1 - self.epsilon + self.epsilon / len(self.A)
				else:
					self.pi[sap[0]][a] = self.epsilon / len(self.A)
			


	# Receive a message form RLGlue
	# Args:
	#	Message (str): the message passed
	# returns:
	#	response (str): The agent's response to the message (optional)
	def agent_message(self, message):
		# Output what the agent thinks is the optimal policy
		if message == "policy":
			pi = {}
			for s in self.pi:
				# Choose the action with the highest probability
				maxP = 0
				for a in self.pi[s]:
					if self.pi[s][a] > maxP:
						maxP = self.pi[s][a]
						pi[s] = a
			return pi

	
	def pi_select(self, pi, s, A):
		rand = np.random.random()
		probs = self.pi.get(s,{a:1/len(A) for a in A})
		val = 0
		for a in A:
			val += probs[a]
			if rand < val:
				return a




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
