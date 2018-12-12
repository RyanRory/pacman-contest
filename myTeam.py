from captureAgents import CaptureAgent
from captureAgents import AgentFactory
from game import Directions
from game import Actions
from util import nearestPoint
import distanceCalculator
import random, util
import datetime
from random import choice
from math import log, sqrt
import time

import sys
sys.path.append('teams/Random-Number/')


def createTeam(firstIndex, secondIndex, isRed,
               first='Attacker', second='Denfender'):
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


#########################################################
#  Base CaptureAgent.                                   #
#  Provide functions used by both attacker and defender #
#########################################################


class BasicAgent(CaptureAgent):
	
	def registerInitialState(self, gameState):
		CaptureAgent.registerInitialState(self, gameState)
		self.distancer.getMazeDistances()
		self.start = gameState.getAgentPosition(self.index)
		self.startFoodNum = len(self.getFood(gameState).asList())
		self.patrolPosition = []
		self.returnPosition = []
		self.isAttacker = False
		self.attacking = True
		self.prefferTop = False
		self.isReversing = False
		self.isMateReversing = False
		self.walls = gameState.getWalls()
		self.topHalfFoods = []
		self.bottomHalfFoods = []
		self.getTopHalfFoods(gameState)
		self.getBottomHalfFoods(gameState)

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

	def isAllEnemiesScared(self, gameState):
		enemies = self.getOpponents(gameState)
		scared = []
		for e in enemies:
			if gameState.getAgentState(e).scaredTimer > 0:
				scared.append(e)
		return len(enemies) == len(scared)

	def getTopHalfFoods(self, gameState):
		self.topHalfFoods = []
		foods = self.getFood(gameState).asList()
		for food in foods:
			x, y = food
			if 1.0*y/gameState.data.layout.height < 0.5:
				self.topHalfFoods.append(food)

	def getBottomHalfFoods(self, gameState):
		self.bottomHalfFoods = []
		foods = self.getFood(gameState).asList()
		for food in foods:
			x, y = food
			if 1.0*y/gameState.data.layout.height >= 0.5:
				self.bottomHalfFoods.append(food)

	def getNearestHome(self, gameState, index):
		currentPosition = gameState.getAgentState(index).getPosition()
		nearestHomeDist = 9999
		nearestHome = None
		for returnPosition in self.returnPosition:
			if self.getMazeDistance(returnPosition, currentPosition) < nearestHomeDist:
				nearestHomeDist = self.getMazeDistance(returnPosition, currentPosition)
				nearestHome = returnPosition

		return nearestHome

	def getExits(self, state):
		legalActions = []
		position = state

		for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
			x, y = position
			dx, dy = Actions.directionToVector(action)
			nextx, nexty = int(x + dx), int(y + dy)
			hitWall = self.walls[nextx][nexty]

			if not hitWall:
				legalActions.append(action)
		return len(legalActions)

	def getReturnPosition(self, gameState):
		width = gameState.data.layout.width
		height = gameState.data.layout.height
		if self.red:
			centralX = (width - 2) / 2
		else:
			centralX = ((width - 2) / 2) + 1
		for i in range(1, height - 1):
			if not gameState.hasWall(centralX, i):
				self.returnPosition.append((centralX, i))
		returnPos = self.returnPosition
		for pos in returnPos:
			if self.getExits(pos) < 2:
				self.returnPosition.remove(pos)

	def getPatrolPosition(self, gameState):
		# TODO:
		width = gameState.data.layout.width
		height = gameState.data.layout.height
		if self.red:
			centralX = (width - 2) / 2
			enemyCentralX = centralX + 1
			for i in range(1, height - 1):
				flag = not gameState.hasWall(enemyCentralX, i)
				for j in range(3):
					flag = flag and not gameState.hasWall(centralX - j, i)
				if flag:
					self.patrolPosition.append((centralX, i))
		else:
			centralX = ((width - 2) / 2) + 1
			enemyCentralX = centralX - 1
			for i in range(1, height - 1):
				flag = not gameState.hasWall(enemyCentralX, i)
				for j in range(3):
					flag = flag and not gameState.hasWall(centralX + j, i)
				if flag:
					self.patrolPosition.append((centralX, i))
		if len(self.patrolPosition) == 0:
			self.patrolPosition = self.returnPosition

		print self.patrolPosition

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

	def getNearestFood(self, gameState, targetFoods = None):
		currentPosition = gameState.getAgentState(self.index).getPosition()
		foods = self.getFood(gameState).asList()
		if targetFoods:
			foods = targetFoods
		# print foods, self.index
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
			if isDefender and not gameState.getAgentState(self.index).isPacman:
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
			# print "WINRATIO 0", datetime.datetime.utcnow()
			if gameState.getAgentState(self.index).numCarrying > 0:
				nearestHome = self.getNearestHome(gameState, self.index)
				minDistance = 9999
				for a, S in moveStates:
					currentPosition = S.getAgentState(self.index).getPosition()
					dist = self.getMazeDistance(currentPosition, nearestHome)
					if dist < minDistance:
						minDistance = dist
						action = a
			else:
				minDistance = 9999
				for a, S in moveStates:
					currentPosition = S.getAgentState(self.index).getPosition()
					dist = min([self.getMazeDistance(currentPosition, food) for food in foodList])
					if dist < minDistance:
						minDistance = dist
						action = a
		return action


###############################################
#   Attacker and Defender Combination.        #
###############################################

class FlexibleAgent(BasicAgent):

	def registerInitialState(self, gameState):
		BasicAgent.registerInitialState(self, gameState)
		self.target = None
		self.foodLastRound = self.getFoodYouAreDefending(gameState).asList()
		self.invaderLastPosition = []
		self.patrolDict = {}
		self.postions = []
		self.matePositions = []
		self.lastPatrol = None
		self.targetFoods = None

		self.getReturnPosition(gameState)
		self.getPatrolPosition(gameState)

		# self.CleanPatrolPostions(gameState.data.layout.height)
		self.distFoodToPatrol(gameState)

	def getTargetFood(self, gameState):
		teammates = self.getTeam(gameState)
		teammates.remove(self.index)
		teammate = teammates[0]
		mateX, mateY = gameState.getAgentState(teammate).getPosition()
		myX, myY = gameState.getAgentState(self.index).getPosition()
		mateInTop = 1.0*mateY/gameState.data.layout.height <= 0.5
		meInTop = 1.0*myY/gameState.data.layout.height <= 0.5
		self.getTopHalfFoods(gameState)
		self.getBottomHalfFoods(gameState)

		if gameState.getAgentState(teammate).isPacman != gameState.getAgentState(self.index).isPacman:
			self.targetFoods = None
			return

		if len(self.bottomHalfFoods) == 0 or len(self.topHalfFoods) == 0:
			self.targetFoods = None
			return

		if len(self.getFood(gameState).asList()) <= 4:
			self.targetFoods = None
			return

		if mateInTop == meInTop:
			if gameState.getAgentState(teammate).isPacman and not gameState.getAgentState(self.index).isPacman:
				if mateInTop:
					self.targetFoods = self.bottomHalfFoods
				else:
					self.targetFoods = self.topHalfFoods
			else:
				if self.prefferTop:
					self.targetFoods = self.topHalfFoods
				else:
					self.targetFoods = self.bottomHalfFoods

		if mateInTop and not meInTop:
			self.targetFoods = self.bottomHalfFoods
			return

		if not mateInTop and meInTop:
			self.targetFoods = self.topHalfFoods

	def judgeReversing(self, gameState):
		self.postions.append(gameState.getAgentState(self.index).getPosition())
		if len(self.postions) >= 4:
			positionSet = set(self.postions)
			# print positionSet, datetime.datetime.utcnow()
			if len(positionSet) == 2:
				self.isReversing = True
				self.postions = list(positionSet)
				# print 'REVERSING!!!!', self.attacking
			else:
				self.isReversing = False
				self.postions.pop(0)
		else:
			self.isReversing = False
			
	def judgeMateReversing(self, gameState):
		teammates = self.getTeam(gameState)
		teammates.remove(self.index)
		self.matePositions.append(gameState.getAgentState(teammates[0]).getPosition())
		if len(self.matePositions) >= 4:
			positionSet = set(self.matePositions)
			# print positionSet, datetime.datetime.utcnow()
			if len(positionSet) == 2:
				self.isMateReversing = True
				self.matePositions = list(positionSet)
			# print 'REVERSING!!!!', self.attacking
			else:
				self.isMateReversing = False
				self.matePositions.pop(0)
		else:
			self.isMateReversing = False

	def getEatenFoods(self, gameState):
		foodEatenPosition = set(self.foodLastRound) - set(self.getFoodYouAreDefending(gameState).asList())
		self.foodLastRound = self.getFoodYouAreDefending(gameState).asList()
		return foodEatenPosition

	def getInvaderPositions(self, gameState):
		enemies = self.getOpponents(gameState)
		invaders = [e for e in enemies if gameState.getAgentState(e).isPacman]
		if len(invaders) == 0:
			# print "NO INVADER", datetime.datetime.utcnow()
			self.invaderLastPosition = []
			return []

		invaderPostions = []
		for i in invaders:
			if gameState.getAgentState(i).getPosition():
				invaderPostions.append(gameState.getAgentState(i).getPosition())
		foodsEatenPositions = self.getEatenFoods(gameState)
		if len(foodsEatenPositions) > 0:
			for pos in foodsEatenPositions:
				if pos not in invaderPostions:
					invaderPostions.append(pos)
		if len(invaderPostions) == 0:
			if len(self.invaderLastPosition) > 0:
				for pos in self.invaderLastPosition:
					if pos not in invaderPostions:
						invaderPostions.append(pos)
		else:
			self.invaderLastPosition = invaderPostions
		# print "INVADER", invaderPostions
		return invaderPostions

	def evaluateEvironment(self, gameState):
		teammates = self.getTeam(gameState)
		teammates.remove(self.index)
		enemies = self.getOpponents(gameState)
		invaders = [e for e in enemies if gameState.getAgentState(e).isPacman]
		currentPosition = gameState.getAgentState(self.index).getPosition()
		myNearestHome = self.getNearestHome(gameState, self.index)
		# self.returnPosition replace self.patrol
		isNearPatrol = min(self.getMazeDistance(currentPosition, p) for p in self.returnPosition) <= 5
		nearestGhostDist, nearestGhost = self.getNearestGhost(gameState)
		mateGhostDist = 9999
		for e in enemies:
			if gameState.getAgentState(e).getPosition():
				d = self.getMazeDistance(gameState.getAgentState(teammates[0]).getPosition(), gameState.getAgentState(e).getPosition())
				if d < mateGhostDist:
					mateGhostDist = d
		invaderPostions = self.getInvaderPositions(gameState)

		foodList = self.getFood(gameState).asList()

		if len(foodList) <= 2 and not gameState.getAgentState(self.index).isPacman:
			return False

		if self.isReversing and self.isMateReversing:
			return not self.attacking

		if gameState.getAgentState(enemies[0]).scaredTimer > 0 and gameState.getAgentState(enemies[1]).scaredTimer > 0 and len(invaderPostions) == 0:
			return True

		if gameState.getAgentState(self.index).scaredTimer > self.getMazeDistance(currentPosition, myNearestHome) and gameState.getAgentState(self.index).isPacman:
			return True

		if len(invaders) > 0 and gameState.getAgentState(self.index).scaredTimer > 0 and not self.attacking:
			if min(self.getMazeDistance(currentPosition, pos) for pos in invaderPostions) > gameState.getAgentState(self.index).scaredTimer/4:
				return self.attacking

		if gameState.getAgentState(self.index).scaredTimer > 0:
			return True

		if len(invaderPostions) > 1:
			return False

		if len(invaders) > 0 and len(invaderPostions) == 0 and gameState.getAgentState(teammates[0]).isPacman and gameState.getAgentState(self.index).isPacman:
			myDistToHome = self.getMazeDistance(currentPosition, myNearestHome)
			mateDistToHome = min(self.getMazeDistance(gameState.getAgentState(t).getPosition(), self.getNearestHome(gameState, t)) for t in teammates)
			if myDistToHome < mateDistToHome:
				return False

		if len(invaderPostions) > 0:
			myDistToInvader = self.getMazeDistance(currentPosition, invaderPostions[0])
			mateDistInvader = self.getMazeDistance(gameState.getAgentState(teammates[0]).getPosition(), invaderPostions[0])
			if gameState.getAgentState(self.index).scaredTimer < myDistToInvader < mateDistInvader:
				return False
			elif gameState.getAgentState(self.index).scaredTimer > myDistToInvader:
				return True

		if nearestGhost is None and isNearPatrol:
			return True

		if nearestGhost and isNearPatrol and not gameState.getAgentState(self.index).isPacman:
			return False

		if gameState.getAgentState(self.index).isPacman != self.isAttacker and len(invaders) > 0:
			if gameState.getAgentState(teammates[0]).isPacman:
				return False
			else:
				return True

		if not self.isAttacker and mateGhostDist < 3 and len(invaders) == 0:
			return True

		return self.attacking

	def getAttackTarget(self, gameState):
		myGhostDist, nearestGhost = self.getNearestGhost(gameState)
		nearestHome = self.getNearestHome(gameState, self.index)
		nearestFoodDist, nearestFood, passBy = self.getNearestFood(gameState, self.targetFoods)
		capsuleDist, nearestCapsule = self.getNearestCapsule(gameState)
		teammates = self.getTeam(gameState)
		teammates.remove(self.index)
		currentPosition = gameState.getAgentState(self.index).getPosition()
		myX, myY = currentPosition
		matePosition = gameState.getAgentState(teammates[0]).getPosition()
		mateX, mateY = matePosition
		width = gameState.data.layout.width
		if self.red:
			centralX = (width - 2) / 2
		else:
			centralX = ((width - 2) / 2) + 1

		'''Strategy: time is not enough'''
		if self.getScore(gameState) < 0 <= gameState.getAgentState(self.index).numCarrying + gameState.getAgentState(teammates[0]).numCarrying + self.getScore(gameState):
			if gameState.data.timeleft - 5 < self.getMazeDistance(currentPosition, nearestHome) < gameState.data.timeleft:
				self.target = nearestHome

		if self.getScore(gameState) == 0 and gameState.getAgentState(self.index).numCarrying + gameState.getAgentState(teammates[0]).numCarrying + self.getScore(gameState) > 0:
			if gameState.data.timeleft - 5 < self.getMazeDistance(currentPosition, nearestHome) < gameState.data.timeleft:
				self.target = nearestHome

		'''Strategy 1: Give up last two foods.'''
		if len(self.getFood(gameState).asList()) <= 2:
			self.target = None
			return

		'''Strategy: If too close to each other and around patrol'''
		if not gameState.getAgentState(self.index).isPacman and not gameState.getAgentState(teammates[0]).isPacman and util.manhattanDistance(currentPosition, matePosition) <= 3 and abs(myX - centralX) <=1 and abs(mateX - centralX) <=1:
			x, y = currentPosition
			if 1.0*y/gameState.data.layout.height < 0.5 != self.prefferTop:
				self.target = max(self.patrolPosition, key=lambda x: util.manhattanDistance(x, matePosition))
				return


		'''Strategy 2: Go home when carry 1/4 foods.'''

		if gameState.getAgentState(self.index).numCarrying >= self.startFoodNum/4 and not self.isAllEnemiesScared(gameState):
			if not (nearestGhost and nearestGhost.scaredTimer > 0 or nearestFoodDist == 1):
				self.target = nearestHome
				return

		'''Strategy 3: Greedy eat foods when safe.'''

		if (nearestGhost is None) or (nearestGhost and myGhostDist >= 6) or (nearestGhost and nearestGhost.scaredTimer > 5):
			# print 'HAHAHAHAHAHAHA', nearestGhost, myGhostDist, datetime.datetime.utcnow()
			if gameState.getAgentState(self.index).numCarrying < self.startFoodNum-2 or passBy or self.isAllEnemiesScared(gameState):
				self.target = nearestFood
				return

		'''Strategy 4: Greedy eat capsule when 1/2 nearestGhostDistance closer than enemy.'''

		if nearestGhost and (not nearestGhost.isPacman) and nearestCapsule and capsuleDist <= myGhostDist/2:
			self.target = nearestCapsule
			return

		'''Strategy 5: Go home when nearest ghost is farther and carrying foods.'''

		if nearestGhost and (not nearestGhost.isPacman) and nearestGhost.scaredTimer <= 0:
			if gameState.getAgentState(self.index).numCarrying > 0:
				nearestHome = self.getNearestHome(gameState, self.index)
				if self.getMazeDistance(nearestHome, gameState.getAgentState(self.index).getPosition()) < self.getMazeDistance(nearestHome, nearestGhost.getPosition()):
					self.target = nearestHome
					return

		self.target = None

	def chooseAttackAction(self, gameState):
		self.getTargetFood(gameState)
		self.getAttackTarget(gameState)

		if self.target:
			if self.isReversing and self.target not in self.postions:
				action = self.aStarSearch(gameState, self.postions[0])
				return action
			nearestFoodDist, nearestFood, passBy = self.getNearestFood(gameState, self.targetFoods)
			if self.target == nearestFood and self.targetFoods is None:
				x, y = gameState.getAgentState(self.index).getPosition()
				if 1.0*y/gameState.data.layout.height <= 0.5 != self.prefferTop:
					teammates = self.getTeam(gameState)
					teammates.remove(self.index)
					action = self.aStarSearch(gameState, gameState.getAgentState(teammates[0]).getPosition())
					return action
			nearestHome = self.getNearestHome(gameState, self.index)
			enemies = self.getOpponents(gameState)
			d = 9999
			nearestGhost = None
			for e in enemies:
				if gameState.getAgentState(e).getPosition():
					if self.getMazeDistance(gameState.getAgentState(e).getPosition(), nearestHome) < d:
						d = self.getMazeDistance(gameState.getAgentState(e).getPosition(), nearestHome)
						nearestGhost = gameState.getAgentState(e)
			if self.target == nearestHome and nearestGhost:
				action = self.aStarSearch(gameState, nearestGhost.getPosition())
				return action
			if gameState.getAgentState(self.index).scaredTimer > 0 and nearestGhost:
				action = self.aStarSearch(gameState, nearestGhost.getPosition())
				return action
			greedyAction = self.getGreedyAction(gameState, self.target)
			return greedyAction
		else:
			if self.target in self.patrolPosition:
				return self.aStarSearch(gameState, None)
			if self.isReversing:
				nearestHome = self.getNearestHome(gameState, self.index)
				self.target = nearestHome
				if self.target not in self.postions:
					action = self.aStarSearch(gameState, self.postions[0])
					return action
				else:
					self.postions.pop(0)
					return Directions.STOP
			if len(self.getFood(gameState).asList()) <= 2:
				nearestHome = self.getNearestHome(gameState, self.index)
				self.target = nearestHome
				return self.aStarSearch(gameState, None)
			uctAction = self.getUCTAction(gameState)
			return uctAction

	def getDefenceTarget(self, gameState):
		currentPosition = gameState.getAgentPosition(self.index)
		teammates = self.getTeam(gameState)
		teammates.remove(self.index)
		teammate = teammates[0]
		invaders = self.getInvaders(gameState)

		if currentPosition == self.target:
			self.target = None
		"""
		if there is invaders: Go for the  nearest invader postion directly
		"""
		invaderPositions = self.getInvaderPositions(gameState)
		if len(invaderPositions) > 0:
			self.target = min(invaderPositions, key=lambda x: self.getMazeDistance(currentPosition, x))
		else:
			self.target = None
		#if len(invaders) > 0:
		# 	invaderPositions = [invader.getPosition() for invader in invaders]
		# 	self.target = min(invaderPositions, key=lambda x: self.getMazeDistance(currentPosition, x))
		# elif self.foodLastRound:
		# 	foodEatenPosition = self.getEatenFoods(gameState)
		# 	if len(foodEatenPosition) > 0:
		# 		self.target = foodEatenPosition.pop()

		if self.target is None and gameState.getAgentState(self.index).isPacman:
			self.target = self.getNearestHome(gameState, self.index)

		"""
		when there are only 5 food dots remaining. defender patrol around these food rather than  boundary line.
		"""
		if self.target is None and len(self.getFoodYouAreDefending(gameState).asList()) <= 5:
			self.target = random.choice(self.getDefendingTarget(gameState))
		elif self.target is None:
			# random to choose a position around the boundary to patrol.
			choices = self.patrolDict.keys()
			if not gameState.getAgentState(teammate).isPacman:
				maxPatrolDist = max(util.manhattanDistance(gameState.getAgentState(teammate).getPosition(), c) for c in choices)
				for c in choices:
					if util.manhattanDistance(gameState.getAgentState(teammate).getPosition(), c) == maxPatrolDist:
						self.target = c
						break
				if currentPosition == self.target:
					self.target = None
			if self.target == None:
				# print "RANDOM PATROL", choices
				if self.lastPatrol and len(choices) > 1:
					choices = filter(lambda x: util.manhattanDistance(x, self.lastPatrol) > 3, choices)
				self.target = random.choice(choices)
			self.lastPatrol = self.target

			# print self.index, self.target, datetime.datetime.utcnow()

	def chooseDefenderAction(self, gameState):
		self.getDefenceTarget(gameState)

		if gameState.getAgentState(self.index).isPacman:
			myGhostDist, nearestGhost = self.getNearestGhost(gameState)
			if nearestGhost:
				# print self.index, nearestGhost, 'DEFENDING'
				return self.aStarSearch(gameState, nearestGhost.getPosition())

		if self.isReversing:
			if self.target not in self.postions:
				return self.aStarSearch(gameState, self.postions[0])
			elif not gameState.getAgentState(self.index).isPacman:
				self.postions.pop(0)
				return Directions.STOP

		return self.getGreedyAction(gameState, self.target, True)

	# Defender functions
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

	def CleanPatrolPostions(self, height):
		while len(self.patrolPosition) > (height - 2) / 2:
			self.patrolPosition.pop(0)
			self.patrolPosition.pop(len(self.patrolPosition) - 1)

	def getDefendingTarget(self, gameState):
		return self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)

	def chooseHelpDefenderAction(self, gameState):
		teammates = self.getTeam(gameState)
		teammates.remove(self.index)
		teammate = teammates[0]
		matePosition = gameState.getAgentState(teammate).getPosition()
		if not gameState.getAgentState(teammate).isPacman:
			return self.chooseDefenderAction(gameState)
		else:
			self.getDefenceTarget(gameState)
			return self.aStarSearch(gameState, matePosition)

	def chooseHelpAttackerAction(self, gameState):
		return self.chooseAttackAction(gameState)

	# -----------------------------------something needed for A*-------------------------#

	def aStarSearch(self, gameState, position, fullPath = False):
		# print "ASTAR", datetime.datetime.utcnow()
		"""Search the node that has the lowest combined cost and heuristic first."""
		currentPosition = gameState.getAgentPosition(self.index)
		path = []

		currentPos = currentPosition
		priorityQueue = util.PriorityQueue()
		cost_so_far = {}
		priorityQueue.push((currentPos, []), 0)
		cost_so_far[currentPos] = 0

		while not priorityQueue.isEmpty():
			currentPos,actionList = priorityQueue.pop()

			if currentPos == self.target:
				path = actionList

			nextMoves = self.getSuccessors(currentPos,actionList,gameState, position)
			for nextNode in nextMoves:
				new_cost = cost_so_far[currentPos] + nextNode[2]
				if nextNode[0][0] not in cost_so_far:
					cost_so_far[nextNode[0][0]] = new_cost
					priorityQueue.push((nextNode[0][0], actionList + [nextNode[1]]),
					                   new_cost + self.heuristic(nextNode[0][0],gameState))
				elif new_cost < cost_so_far[nextNode[0][0]]:
					priorityQueue.push((nextNode[0][0], actionList + [nextNode[1]]),
					                   new_cost + self.heuristic(nextNode[0][0],gameState))

		if fullPath:
			return path

		if len(path) > 0:
			return path[0]
		else:
			print "RAMDOM ASTAR"
			return Directions.STOP
			# return random.choice(gameState.getLegalActions(self.index))

	def XY(self,path,currentState):
		pathxy = []
		pathxy.append(currentState)

		for action in path:
			x, y = currentState
			dx, dy = Actions.directionToVector(action)
			nextx, nexty = int(x + dx), int(y + dy)
			pathxy.append((nextx,nexty))
			currentState = (nextx,nexty)

		return pathxy

	def getGhosts(self,gameState):
		enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
		ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
		return ghosts

	def getCostFunction(self, nextState, gameState, position):
		agentState = gameState.data.agentStates[self.index]
		isPacman = agentState.isPacman
		cost = 0
		invaders = self.getInvaders(gameState)
		ghosts = self.getGhosts(gameState)
		invaderPos = [a.getPosition() for a in invaders]
		ghostPos = []
		# for a in ghosts:
		# 	x, y = a.getPosition()
		# 	ghostPos += [(x, y)]
		# 	ghostPos += [(x, y), (x+1, y), (x-1, y), (x, y+1), (x, y-1)]
		# scaredGhostPos = []
		# for index in self.getOpponents(gameState):
		# 	if gameState.getAgentState(index).scaredTimer > 0:
		# 		if gameState.getAgentState(index).getPosition():
		# 			x, y = gameState.getAgentState(index).getPosition()
		# 			scaredGhostPos += [(x, y), (x+1, y), (x-1, y), (x, y+1), (x, y-1)]

		ghostPos = [a.getPosition() for a in ghosts]
		scaredGhostPos = []

		for index in self.getOpponents(gameState):
			if gameState.getAgentState(index).scaredTimer > 0 and gameState.getAgentState(index).getPosition():
				scaredGhostPos.append(gameState.getAgentState(index).getPosition())

		if isPacman:
			if nextState in ghostPos:
				if nextState not in scaredGhostPos:
					cost = 9999
				else:
					cost = 1
			else:
				cost = 1
		else:
			if nextState in invaderPos:
				cost = 0
			else:
				cost = 1
			if not self.attacking:
				nextState_x, nextState_y = nextState
				if gameState.isOnRedTeam(self.index):
					midWidth = int(gameState.data.layout.width/2)
					if nextState_x >= midWidth:
						cost += 9999
				else:
					midWidth = int(gameState.data.layout.width/2)+1
					if nextState_x <= midWidth:
						cost += 9999

		if position and nextState == position:
			cost += 9999

		return cost

	def getSuccessors(self, state, actionList, gameState, pos):
		successors = []
		position = state

		for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
			x, y = position
			dx, dy = Actions.directionToVector(action)
			nextx, nexty = int(x + dx), int(y + dy)
			hitWall = self.walls[nextx][nexty]

			if not hitWall:
				nextState = (nextx, nexty)
				cost = self.getCostFunction(nextState, gameState, pos)
				successors.append(((nextState, actionList), action, cost))
		return successors

	def heuristic(self, state, gameState):

		h = {}
		hValue = 0

		goalPosition = self.target

		if self.manhattonDistance(state,goalPosition) <=6:
			hValue = self.getMazeDistance(state,goalPosition)
		else:
			hValue = self.manhattonDistance(state,goalPosition)

		return hValue

	def manhattonDistance(self, p1, p2):
		x1, y1 = p1
		x2, y2 = p2
		distance = abs(x2 - x1) + abs(y2 - y1)
		return distance

	#-----------------------------------something needed for A*-------------------------#



class Attacker(FlexibleAgent):
	def registerInitialState(self, gameState):
		FlexibleAgent.registerInitialState(self, gameState)
		self.isAttacker = True
		self.attacking = True
		self.prefferTop = False
		self.targetFoods = self.bottomHalfFoods
		self.distFoodToPatrol(gameState)
		print self.index, self.isAttacker

	def chooseAction(self, gameState):
		start = time.time()
		self.judgeReversing(gameState)
		self.judgeMateReversing(gameState)
		# if self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), self.start) < 2:
		# 	print "DEAD", datetime.datetime.utcnow()
		self.attacking = self.evaluateEvironment(gameState)
		# print self.index, self.targetFoods
		# if self.isAttacker != self.attacking:
		# 	print self.index, self.isAttacker, self.attacking
		if self.attacking:
			# preparing for defence
			action = self.chooseAttackAction(gameState)
		else:
			action = self.chooseHelpDefenderAction(gameState)
		print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
		return action


class Denfender(FlexibleAgent):
	def registerInitialState(self, gameState):
		FlexibleAgent.registerInitialState(self, gameState)
		self.isAttacker = False
		self.attacking = True
		self.prefferTop = True
		self.targetFoods = self.topHalfFoods
		# self.CleanPatrolPostions(gameState.data.layout.height)
		self.distFoodToPatrol(gameState)
		print self.index, self.isAttacker

	def chooseAction(self, gameState):
		# start = time.time()
		self.judgeReversing(gameState)
		self.judgeMateReversing(gameState)
		# if gameState.getAgentState(self.index).getPosition() == self.start:
		# 	print "DEAD", datetime.datetime.utcnow()
		self.attacking = self.evaluateEvironment(gameState)
		# print self.index, self.targetFoods
		# if self.isAttacker != self.attacking:
		# 	print self.index, self.isAttacker, self.attacking
		if self.attacking:
			# preparing for attack
			action = self.chooseHelpAttackerAction(gameState)
		else:
			action = self.chooseDefenderAction(gameState)

		# print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
		return action


##############################################################################
#     MCT applied UCB policy. Used to generate and return UCT move           #
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