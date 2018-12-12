from captureAgents import CaptureAgent
from captureAgents import AgentFactory
from game import Directions
from util import nearestPoint
import distanceCalculator
import random, util
import datetime
from random import choice
from math import log, sqrt

import sys
sys.path.append('teams/Random-Number/')


def createTeam(firstIndex, secondIndex, isRed,
               first='Defender', second='Attacker'):
	"""
	This function should return a list of two agents that will form the
	team, initialized using firstIndex and secondIndex as their agent
	index numbers.  isRed is True if the red team is being created, and
	will be False if the blue team is being created.
	As a potentially helpful development aid, this function can take
	additional string-valued keyword arguments ("first" and "second" are
	such arguments in the case of this function), which will come from
	the --redOpts and --blueOpts command-line arguments to capture.py.
	For the nightly contest, however, your team will be created without
	any extra arguments, so you should make sure that the default
	behavior is what you want for the nightly contest.
	"""
	return [eval(first)(firstIndex), eval(second)(secondIndex)]


class MonteCarloFactory(AgentFactory):
	def __init__(self, isRed):
		AgentFactory.__init__(self, isRed)
		self.agentList = ['attacker', 'defender']

	def getAgent(self, index):
		if len(self.agentList) > 0:
			agent = self.agentList.pop(0)
			if agent == 'attacker':
				return Attacker(index)
		return Defender(index)


#########################################################
#  Base CaptureAgent.                                   #
#  Provide functions used by both attacker and defender #
#########################################################


class BasicAgent(CaptureAgent):
	
	def registerInitialState(self, gameState):
		CaptureAgent.registerInitialState(self, gameState)
		self.distancer.getMazeDistances()
		self.start = gameState.getAgentPosition(self.index)
		self.patrolPosition = []

	def getPatrolPosition(self, gameState):
		width = gameState.data.layout.width
		height = gameState.data.layout.height
		if self.red:
			centralX = (width - 2) / 2
		else:
			centralX = ((width - 2) / 2) + 1
		for i in range(1, height - 1):
			if not gameState.hasWall(centralX, i):
				self.patrolPosition.append((centralX, i))

	def getSuccessor(self, gameState, action):
		"""
		Find the next successor which is a grid position(x, y).
		"""
		successor = gameState.generateSuccessor(self.index, action)
		position = successor.getAgentState(self.index).getPosition()
		if position != nearestPoint(position):
			return successor.generateSuccessor(self.index, action)
		else:
			return successor

	def getNearestGhost(self, gameState):
		currentPosition = gameState.getAgentState(self.index).getPosition()
		ghosts = []
		ghostsInRange = []
		for index in self.getOpponents(gameState):
			ghosts.append(gameState.getAgentState(index))
		for ghost in ghosts:
			if ghost.getPosition() and util.manhattanDistance(currentPosition, ghost.getPosition()) <= 5:
				ghostsInRange.append(ghost)

		if len(ghostsInRange) > 0:
			distances = []
			for ghost in ghostsInRange:
				distances.append((self.getMazeDistance(currentPosition, ghost.getPosition()), ghost))
			return min(distances)
		return None, None

	def getNearestFood(self, gameState):
		currentPosition = gameState.getAgentState(self.index).getPosition()
		foods = self.getFood(gameState).asList()
		oneStepFoods = [food for food in foods if self.getMazeDistance(currentPosition, food) == 1]
		twoStepFoods = [food for food in foods if self.getMazeDistance(currentPosition, food) == 2]
		passBy = (len(oneStepFoods) == 1 and len(twoStepFoods) == 0)
		if len(foods) > 0:
			distances = []
			for food in foods:
				distances.append((self.getMazeDistance(currentPosition, food), food, passBy))
			return min(distances)
		return None, None, None

	def getNearestCapsule(self, gameState):
		currentPosition = gameState.getAgentState(self.index).getPosition()
		capsules = self.getCapsules(gameState)
		if len(capsules) > 0:
			distances = []
			for capsule in capsules:
				distances.append((self.getMazeDistance(currentPosition, capsule), capsule))
			return min(distances)
		return None, None

	def getBFSAction(self, gameState, goal):
		"""
		BFS Algorithm to help ally chase enemy.
		"""
		queue = util.Queue()
		visited = []
		path = []
		queue.push((gameState, path))

		while not queue.isEmpty():
			currState, path = queue.pop()
			visited.append(currState)
			if currState.getAgentState(self.index).getPosition() == goal:
				return path[0]
			actions = currState.getLegalActions(self.index)
			actions.remove(Directions.STOP)
			for action in actions:
				nextState = currState.generateSuccessor(self.index, action)
				if not nextState in visited:
					visited.append(nextState)
					queue.push((nextState, path + [action]))
		return []

	def getGreedyAction(self, gameState, goal, isDefender = False):
		"""
		Greedy Algorithm to eat nearest goal.
		"""
		actions = gameState.getLegalActions(self.index)
		actions.remove(Directions.STOP)
		goodActions = []
		values = []
		for action in actions:
			nextState = gameState.generateSuccessor(self.index, action)
			if isDefender:
				if not nextState.getAgentState(self.index).isPacman:
					nextPostion = nextState.getAgentPosition(self.index)
					goodActions.append(action)
					values.append(self.getMazeDistance(nextPostion, goal))
			else:
				nextPostion = nextState.getAgentPosition(self.index)
				goodActions.append(action)
				values.append(self.getMazeDistance(nextPostion, goal))

		# Randomly chooses between ties.
		best = min(values)
		ties = [combine for combine in zip(values, goodActions) if combine[0] == best]

		return random.choice(ties)[1]

	def getUCTAction(self, gameState):
		"""
		UCT algorithm to choose which action is more rational.
		"""

		action, winRatio = MCTNode(gameState, self.index).getUctAction()

		actions = gameState.getLegalActions(self.index)
		actions.remove(Directions.STOP)
		moveStates = [(a, gameState.generateSuccessor(self.index, a)) for a in actions]

		foodList = self.getFood(gameState).asList()

		if len(foodList) <= 2:
			bestDist = 9999
			for a in actions:
				successor = gameState.generateSuccessor(self.index, a)
				nextPostion = successor.getAgentPosition(self.index)
				dist = self.getMazeDistance(self.start, nextPostion)
				if dist < bestDist:
					bestAction = a
					bestDist = dist
			return bestAction

		# if percentage of win is 0, go and eat the nearest food
		if winRatio == 0.0:
			minDistance = 9999
			for a, S in moveStates:
				currentPosition = S.getAgentState(self.index).getPosition()
				dist = min([self.getMazeDistance(currentPosition, food) for food in foodList])
				if dist < minDistance:
					minDistance = dist
					action = a
		return action

##############################
#   Attacker Agent.          #
##############################


class Attacker(BasicAgent):

	def registerInitialState(self, gameState):
		BasicAgent.registerInitialState(self, gameState)
		self.distancer.getMazeDistances()
		self.start = gameState.getAgentPosition(self.index)
		self.getPatrolPosition(gameState)

	def getNearestHome(self, gameState):
		currentPosition = gameState.getAgentState(self.index).getPosition()
		nearestHomeDist = 9999
		nearestHome = None
		for returnPosition in self.patrolPosition:
			if self.getMazeDistance(returnPosition, currentPosition) < nearestHomeDist:
				nearestHomeDist = self.getMazeDistance(returnPosition, currentPosition)
				nearestHome = returnPosition

		return nearestHome

	def chooseAction(self, gameState):
		"""
		Choose next action according to strategy.
		"""

		myGhostDist, nearestGhost = self.getNearestGhost(gameState)
		capsuleDist, nearestCapsule = self.getNearestCapsule(gameState)

		'''Strategy 1: Give up last two foods.'''
		if len(self.getFood(gameState).asList()) <= 2:
			uctAction = self.getUCTAction(gameState)
			return uctAction

		'''Strategy 2: BFS eat enemy when ally not around.'''
		if nearestGhost and nearestGhost.isPacman and gameState.getAgentState(self.index).scaredTimer == 0:
			teammates = self.getTeam(gameState)
			teammates.remove(self.index)
			mateGhostDist = min(self.getMazeDistance(nearestGhost.getPosition(), gameState.getAgentState(t).getPosition()) for t in teammates)
			if mateGhostDist > myGhostDist:
				helpAction = self.getGreedyAction(gameState, nearestGhost.getPosition())
				return helpAction

		'''Strategy 3: Greedy eat foods when safe.'''

		if (nearestGhost is None) or (nearestGhost and myGhostDist >= 6) or (nearestGhost and nearestGhost.scaredTimer > 5):
			nearestFoodDist, nearestFood, passBy = self.getNearestFood(gameState)
			if gameState.getAgentState(self.index).numCarrying < 6 or passBy:
				return self.getGreedyAction(gameState, nearestFood)

		'''Strategy 4: Greedy eat capsule when half nearestGhostDistance closer than enemy.'''

		if nearestGhost and (not nearestGhost.isPacman) and nearestCapsule and capsuleDist <= myGhostDist / 2:
			return self.getGreedyAction(gameState, nearestCapsule)


		if (nearestGhost is None) or (nearestGhost and myGhostDist >= 6) or (nearestGhost and nearestGhost.scaredTimer > 5):
			if gameState.getAgentState(self.index).numCarrying >= 5:
				nearestHome = self.getNearestHome(gameState)
				return self.getGreedyAction(gameState, nearestHome)

		'''Strategy 5: other situations use UCT algorithm to trade off: invader, score, escape and eat capsule.'''

		uctAction = self.getUCTAction(gameState)
		return uctAction

##############################
#   Defender Agent.          #
##############################


class Defender(BasicAgent):
	
	def registerInitialState(self, gameState):
		BasicAgent.registerInitialState(self, gameState)
		self.target = None
		self.foodLastRound = None
		self.patrolDict = {}
		
		self.distancer.getMazeDistances()
		self.getPatrolPosition(gameState)
		self.CleanPatrolPostions(gameState.data.layout.height)
		self.distFoodToPatrol(gameState)

	def distFoodToPatrol(self, gameState):
		defendingFoods = self.getFoodYouAreDefending(gameState).asList()
		# Get the min distance from the food to patrol points.
		for position in self.patrolPosition:
			closestFoodDist = 9999
			for food in defendingFoods:
				dist = self.getMazeDistance(position, food)
				if dist < closestFoodDist:
					closestFoodDist = dist
			self.patrolDict[position] = closestFoodDist

	"""
		Remove some of patrol positions. 
		when the size (height) of maze is greater than 18,  leave 5 position to be un-patrolled
		when the size of maze is less than 18, leave half of the postions patrolled.
		:param height: height of the maze
	"""

	def CleanPatrolPostions(self, height):
		while len(self.patrolPosition) > (height - 2) / 2:
			self.patrolPosition.pop(0)
			self.patrolPosition.pop(len(self.patrolPosition) - 1)

	"""
	Update the minimum distance between patrol points to closest food
	"""

	def updateMiniFoodDistance(self, gameState):
		if self.foodLastRound and len(self.foodLastRound) != len(self.getFoodYouAreDefending(gameState).asList()):
			self.distFoodToPatrol(gameState)

	def getInvaders(self, gameState):
		enemies = []
		for index in self.getOpponents(gameState):
			enemies.append(gameState.getAgentState(index))
		result = []
		for enemy in enemies:
			if enemy.isPacman and enemy.getPosition():
				result.append(enemy)
		return result
		# return filter(lambda x: x.isPacman and x.getPosition() != None, enemies)

	def getDefendingTarget(self, gameState):
		return self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)

	def chooseAction(self, gameState):
		self.updateMiniFoodDistance(gameState)
		currentPosition = gameState.getAgentPosition(self.index)
		invaders = self.getInvaders(gameState)

		if currentPosition == self.target:
			self.target = None

		"""
		if there is invaders: Go for the  nearest invader postion directly
		"""
		if len(invaders) > 0:
			# print invaders
			invaderPositions = [invader.getPosition() for invader in invaders]
			# print invaderPositions
			self.target = min(invaderPositions, key=lambda x: self.getMazeDistance(currentPosition, x))
		elif self.foodLastRound:
			foodEatenPosition = set(self.foodLastRound) - set(self.getFoodYouAreDefending(gameState).asList())
			if len(foodEatenPosition) > 0:
				self.target = foodEatenPosition.pop()

		# record the food list. it will be compared to next round's food list to determine opponent's position
		self.foodLastRound = self.getFoodYouAreDefending(gameState).asList()

		"""
		when there are only 5 food dots remaining. defender patrol around these food rather than  boundary line.
		"""
		if self.target is None and len(self.getFoodYouAreDefending(gameState).asList()) <= 5:
			self.target = random.choice(self.getDefendingTarget(gameState))
		elif self.target is None:
			# random to choose a position around the boundary to patrol.
			self.target = random.choice(self.patrolDict.keys())

		return self.getGreedyAction(gameState, self.target, True)

##############################################################################
#  MCT applied UCB1 policy. Used to generate and return UCT move             #
#  Developed based on Jeff Bradberry's board game:                           #
#  https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/ #
##############################################################################


class MCTNode:
	def __init__(self, gameState, playerIndex, **kwargs):
		# define the time allowed for simulation
		# seconds can be 0.8? Cuz the whole waiting time is 1.
		seconds = kwargs.get('time', 0.6)
		# seconds = kwargs.get('time', 0.8)
		self.calculateTime = datetime.timedelta(seconds=seconds)

		self.maxMoves = kwargs.get('maxMoves', 20)
		# self.maxMoves = kwargs.get('maxMoves', 30)
		self.states = [gameState]
		self.index = playerIndex
		self.wins = util.Counter()
		self.plays = util.Counter()
		self.C = kwargs.get('C', 1)
		self.distancer = distanceCalculator.Distancer(gameState.data.layout)
		self.distancer.getMazeDistances()
		self.gameState = gameState

		if gameState.isOnRedTeam(self.index):
			self.enemies = gameState.getBlueTeamIndices()
			self.foodList = gameState.getBlueFood()
			self.capsule = gameState.getBlueCapsules()
		else:
			self.enemies = gameState.getRedTeamIndices()
			self.foodList = gameState.getRedFood()
			self.capsule = gameState.getRedCapsules()

	def getMazeDistance(self, pos1, pos2):
		"""
		Returns the distance between two points; These are calculated using the provided distancer object.
		If distancer.getMazeDistances() has been called, then maze distances are available.
		Otherwise, this just returns Manhattan distance.
		"""
		dist = self.distancer.getDistance(pos1, pos2)
		return dist

	def takeToEmptyAlley(self, gameState, action, depth):
		"""
		Verify if an action takes the agent to an alley with no pacdots.
		"""
		preScore = gameState.getScore()
		nextState = gameState.generateSuccessor(self.index, action)
		postScore = nextState.getScore()
		if preScore < postScore or depth == 0:
			return False
		actions = nextState.getLegalActions(self.index)
		actions.remove(Directions.STOP)
		reverse = Directions.REVERSE[nextState.getAgentState(self.index).configuration.direction]
		if reverse in actions:
			actions.remove(reverse)
		if len(actions) == 0:
			return True
		for action in actions:
			if not self.takeToEmptyAlley(nextState, action, depth - 1):
				return False
		return True

	def getUctAction(self):
		# get the best move from the
		# current game state and return it.
		state = self.states[-1]
		actions = state.getLegalActions(self.index)
		actions.remove(Directions.STOP)

		for action in actions:
			if self.takeToEmptyAlley(self.gameState, action, 6):
				actions.remove(action)

		# return the action early if there is no other choice
		if not actions:
			return
		if len(actions) == 1:
			return actions[0], 0.0

		games = 0

		begin = datetime.datetime.utcnow()
		while datetime.datetime.utcnow() - begin < self.calculateTime:
			self.uctSimulation()
			games += 1

		moveStates = [(action, state.generateSuccessor(self.index, action)) for action in actions]

		winRatio, move = max((float(
			self.wins.get(S.getAgentState(self.index).getPosition(), 0)) / float(
			self.plays.get(S.getAgentState(self.index).getPosition(), 1)), a) for a, S in moveStates)

		return move, winRatio

	def uctSimulation(self):
		# simulate the moves from the current game state
		# and updates self.plays and self.wins
		stateCopy = self.states[:]
		state = stateCopy[-1]
		statesPath = [state]

		# get the ghost and invaders the agent can see at the current game state
		enemies = [state.getAgentState(i) for i in self.enemies if state.getAgentState(i).scaredTimer < 6]
		ghosts = [enemy for enemy in enemies if enemy.getPosition() and not enemy.isPacman]
		invaders = [enemy for enemy in enemies if enemy.isPacman]

		c, d = state.getAgentState(self.index).getPosition()
		currentScore = state.getScore()

		expand = True
		for i in xrange(1, self.maxMoves + 1):
			state = stateCopy[-1]
			# make i evaluates lazily
			actions = state.getLegalActions(self.index)
			actions.remove(Directions.STOP)

			# Bail out early if there is no real choice to be made.
			if not actions:
				return

			moveStates = [(action, state.generateSuccessor(self.index, action)) for action in actions]

			# check if all the results in the actions are in the plays dictionary
			# if they are, use UBT1 to make choice
			if all(self.plays.get(S.getAgentState(self.index).getPosition()) for a, S in moveStates):

				# the number of times state has been visited.
				if self.plays[state.getAgentState(self.index).getPosition()] == 0.0:
					logTotal = 0.5

				else:
					logTotal = float(
						2.0 * log(self.plays[state.getAgentState(self.index).getPosition()]))

				value, move, nstate = max(
					((float(self.wins[S.getAgentState(self.index).getPosition()]) / float(
						self.plays[S.getAgentState(self.index).getPosition()])) +
					 2 * self.C * sqrt(
						logTotal / float(self.plays[S.getAgentState(self.index).getPosition()])), a, S)
					for a, S in moveStates
				)
			else:
				# if not, make a random choice
				move, nstate = choice(moveStates)

			stateCopy.append(nstate)
			statesPath.append(nstate)

			if expand and nstate.getAgentState(self.index).getPosition() not in self.plays:
				# expand the tree
				expand = False
				self.plays[nstate.getAgentState(self.index).getPosition()] = 0.0
				self.wins[nstate.getAgentState(self.index).getPosition()] = 0.0

			'''
			if len(invaders) != 0:
				# if see a invader and ate it, win +1
				ate = False
				for a in invaders:
					if nstate.getAgentState(self.index).getPosition() == a.getPosition():
						ate = True
						break
				if ate:
					# record number of wins
					for s in statesPath:
						if s.getAgentState(self.index).getPosition() not in self.plays:
							continue
						self.wins[s.getAgentState(self.index).getPosition()] += 1.0
						# print self.index, "EAT GHOST +1"
					break
			'''

			x, y = nstate.getAgentState(self.index).getPosition()

			if len(ghosts) > 0:
				currentDistanceToGhost, a = min([(self.getMazeDistance((c, d), g.getPosition()), g) for g in ghosts])
				if util.manhattanDistance((c, d), a.getPosition()) < 6:
					nextDistanceToGhost = min((self.getMazeDistance((x, y), g.getPosition()) for g in ghosts))

					if nextDistanceToGhost < currentDistanceToGhost:
						break

					if nextDistanceToGhost - currentDistanceToGhost > 3 and abs(nstate.getScore() - currentScore) > 0:
						# record number of wins
						for s in statesPath:
							if s.getAgentState(self.index).getPosition() not in self.plays:
								continue
							self.wins[s.getAgentState(self.index).getPosition()] += 1.0
						break

					if nextDistanceToGhost - currentDistanceToGhost > 4:
						# record number of wins
						for s in statesPath:
							if s.getAgentState(self.index).getPosition() not in self.plays:
								continue
							self.wins[s.getAgentState(self.index).getPosition()] += 0.7
						break

			if len(self.capsule) != 0:
				distanceToCapsule, cap = min([(self.getMazeDistance((x, y), cap), cap) for cap in self.capsule])

				if nstate.getAgentState(self.index).getPosition() == cap:
					# record number of wins
					for s in statesPath:
						if s.getAgentState(self.index).getPosition() not in self.plays:
							continue
						self.wins[s.getAgentState(self.index).getPosition()] += 0.002
					break

			if abs(nstate.getScore() - currentScore) > 3:
				# record number of wins
				for s in statesPath:
					if s.getAgentState(self.index).getPosition() not in self.plays:
						continue
					self.wins[s.getAgentState(self.index).getPosition()] += 0.4
				break

		for s in statesPath:
			# record number of plays
			if s.getAgentState(self.index).getPosition() not in self.plays:
				continue
			self.plays[s.getAgentState(self.index).getPosition()] += 1.0