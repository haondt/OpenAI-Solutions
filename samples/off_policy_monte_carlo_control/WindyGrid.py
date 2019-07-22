# Environment for a windy 4/8/9 action gridworld
# Noah Burghardt 2018/10/27
import numpy as np

class GridWorld:
	"""
	Defines the interface of an RLGlue environment

	ie. These methods must be defined in your own environment classes
	"""
	# reward is -1 on all transitions
	# trying to leave the grid doesnt move the agent anywhere 
	# but still rewards -1
	# actions are up, down, right, left
	height = None
	width = None
	start = None
	goal = None
	wind = None
	location = None

	def __init__(self):
		"""
		(run on initialization)
		Declare environment variables.
		"""
		self.height = 7
		self.width = 10
		self.start = None
		self.goal = None
		self.wind = None
		self.location = None

	def env_init(self):
		"""
		Initialize environment variables.
		(run once in experiment)
		"""
		self.start = [0,3]
		self.goal = [7,3]
		# amount of vertical upwards wind shift for each column
		#self.wind = [0,0,0,1,1,1,2,2,1,0]
		self.wind =  [0,0,0,0,0,0,0,0,0,0]
		self.location = self.start.copy()

	def env_start(self):
		"""
		(run at the beginning of each episode)
		The first method called when the experiment starts, called before the
		agent starts.

		Returns:
			The first state observation from the environment.
		"""
		self.location = self.start.copy()
		return tuple(self.location)

	def env_step(self, action):
		"""
		A step taken by the environment.

		Args:
			action: The action taken by the agent

		Returns:
			(float, state, Boolean): a tuple of the reward, state observation,
				and boolean indicating if it's terminal.
		"""
		# save old location
		oldlocation = self.location.copy()
		# add wind
		self.location[1] = self.location[1] + self.wind[self.location[0]]

			
		# check validity of wind action
		if	self.location[0] < 0 or \
			self.location[0] >= self.width or \
			self.location[1] < 0 or \
			self.location[1] >= self.height :

			# action is invalid
			self.location = oldlocation

		# save new old location
		oldlocation = self.location.copy()
		# perform action
		if action == 'up':
			self.location[1] += 1
		elif action == 'down':
			self.location[1] -= 1
		elif action == 'left':
			self.location[0] -= 1
		elif action == 'right':
			self.location[0] += 1
		elif action == 'upleft':
			self.location[0] -= 1
			self.location[1] += 1
		elif action == 'upright':
			self.location[0] += 1
			self.location[1] += 1
		elif action == 'downleft':
			self.location[0] -= 1
			self.location[1] -= 1
		elif action == 'downright':
			self.location[0] += 1
			self.location[1] -= 1
		elif not action == 'stop':
			raise ValueError('invalid action: ' + str(action))
		
		# check validity of action
		if	self.location[0] < 0 or \
			self.location[0] >= self.width or \
			self.location[1] < 0 or \
			self.location[1] >= self.height :

			# action is invalid
			self.location = oldlocation
	
		# check if action was terminal
		terminal = self.location == self.goal
		return -1.0, tuple(self.location.copy()), terminal

	def env_message(self, message):
		"""
		receive a message from RLGlue
		Args:
		   message (str): the message passed
		Returns:
		   str: the environment's response to the message (optional)
		"""
	def get_actions(self):
		return [
			'up',
			'down',
			'left',
			'right',
			'upleft',
			'upright',
			'downleft',
			'downright',
			'stop'
		]
