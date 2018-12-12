from captureAgents import CaptureAgent
from captureAgents import AgentFactory
from game import Directions
from game import Agent
from game import Actions
from util import nearestPoint
import distanceCalculator
import random, util,time
import datetime
from random import choice
from math import log, sqrt

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
        self.patrolPosition = []
        self.isAttacker = False
        self.walls = gameState.getWalls()

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

    def getNearestHome(self, gameState, index):
        currentPosition = gameState.getAgentState(index).getPosition()
        nearestHomeDist = 9999
        nearestHome = None
        for returnPosition in self.patrolPosition:
            if self.getMazeDistance(returnPosition, currentPosition) < nearestHomeDist:
                nearestHomeDist = self.getMazeDistance(returnPosition, currentPosition)
                nearestHome = returnPosition

        return nearestHome

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

    def getNearestFood(self, gameState, targetFoods=None):
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

    def getGreedyAction(self, gameState, goal, isDefender=False):
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
        self.foodLastRound = None
        self.patrolDict = {}
        self.moves = []
        self.lastPatrol = None
        self.targetFoods = None

        self.getPatrolPosition(gameState)
        self.CleanPatrolPostions(gameState.data.layout.height)
        self.distFoodToPatrol(gameState)

    def isReverseDirection(self, d1, d2):
        if d1 == Directions.SOUTH and d2 == Directions.NORTH:
            return True
        if d1 == Directions.NORTH and d2 == Directions.SOUTH:
            return True
        if d1 == Directions.WEST and d2 == Directions.EAST:
            return True
        if d1 == Directions.EAST and d2 == Directions.WEST:
            return True
        return False

    def getTargetFood(self, gameState):
        teammates = self.getTeam(gameState)
        teammates.remove(self.index)
        teammate = teammates[0]
        mateX, mateY = gameState.getAgentState(teammate).getPosition()

        if not gameState.getAgentState(teammate).isPacman:
            self.targetFoods = None
            return

        foods = self.getFood(gameState).asList()
        self.targetFoods = []
        if mateY / gameState.data.layout.height < 0.5:
            for food in foods:
                x, y = food
                if y / gameState.data.layout.height >= 0.5:
                    self.targetFoods.append(food)
        else:
            for food in foods:
                x, y = food
                if y / gameState.data.layout.height < 0.5:
                    self.targetFoods.append(food)

    def isReversing(self):
        reversing = True
        if len(self.moves) > 6:
            for i in range(len(self.moves) - 1):
                reversing = reversing and self.isReverseDirection(self.moves[len(self.moves) - 2 - i],
                                                                  self.moves[len(self.moves) - 1 - i])
        else:
            reversing = False
        return reversing

    def evaluateEvironment(self, gameState):
        teammates = self.getTeam(gameState)
        teammates.remove(self.index)
        enemies = self.getOpponents(gameState)
        currentPosition = gameState.getAgentState(self.index).getPosition()
        myNearestHome = self.getNearestHome(gameState, self.index)
        invaders = [e for e in enemies if gameState.getAgentState(e).isPacman]
        isNearPatrol = min(self.getMazeDistance(currentPosition, p) for p in self.patrolPosition) < 2
        mateGhostDist = 9999
        for e in enemies:
            if gameState.getAgentState(e).getPosition():
                d = self.getMazeDistance(gameState.getAgentState(teammates[0]).getPosition(),
                                         gameState.getAgentState(e).getPosition())
                if d < mateGhostDist:
                    mateGhostDist = d

        if gameState.getAgentState(enemies[0]).scaredTimer > 0 and len(invaders) == 0:
            return True

        if gameState.getAgentState(self.index).scaredTimer > self.getMazeDistance(currentPosition, myNearestHome):
            return True

        if gameState.getAgentState(self.index).isPacman != self.isAttacker:
            if gameState.getAgentState(teammates[0]).isPacman:
                return False
            else:
                return True

        if max(util.manhattanDistance(currentPosition, gameState.getAgentState(t).getPosition()) for t in
               teammates) < 3:
            return self.isAttacker

        if len(self.getInvaders(gameState)) == 0 and isNearPatrol:
            return True

        if not self.isAttacker and mateGhostDist < 3 and len(invaders) == 0:
            return True

        if len(invaders) > 1:
            return False

        for e in invaders:
            if gameState.getAgentState(e).getPosition() is None:
                myDistToHome = self.getMazeDistance(currentPosition, myNearestHome)
                mateDistToHome = min(
                    self.getMazeDistance(gameState.getAgentState(t).getPosition(), self.getNearestHome(gameState, t))
                    for t in teammates)
                if myDistToHome <= mateDistToHome:
                    return False
            else:
                myDistToInvader = self.getMazeDistance(currentPosition, gameState.getAgentState(e).getPosition())
                mateDistInvader = min(self.getMazeDistance(gameState.getAgentState(t).getPosition(),
                                                           gameState.getAgentState(e).getPosition()) for t in teammates)
                if myDistToInvader <= mateDistInvader:
                    return False

        return self.isAttacker

    def getAttackTarget(self, gameState):
        myGhostDist, nearestGhost = self.getNearestGhost(gameState)
        capsuleDist, nearestCapsule = self.getNearestCapsule(gameState)

        '''Strategy 1: Give up last two foods.'''
        if len(self.getFood(gameState).asList()) <= 2:
            self.target = None
            return

        '''Strategy 3: Greedy eat foods when safe.'''

        if (nearestGhost is None) or (nearestGhost and myGhostDist >= 6) or (
                nearestGhost and nearestGhost.scaredTimer > 5):
            nearestFoodDist, nearestFood, passBy = self.getNearestFood(gameState, self.targetFoods)
            if gameState.getAgentState(self.index).numCarrying < 6 or passBy:
                self.target = nearestFood
                return

        '''Strategy 4: Greedy eat capsule when 1/2 nearestGhostDistance closer than enemy.'''

        if nearestGhost and (not nearestGhost.isPacman) and nearestCapsule and capsuleDist <= myGhostDist / 2:
            self.target = nearestCapsule
            return

        if (nearestGhost is None) or (nearestGhost and myGhostDist >= 6) or (
                nearestGhost and nearestGhost.scaredTimer > 5):
            if gameState.getAgentState(self.index).numCarrying >= 5:
                nearestHome = self.getNearestHome(gameState, self.index)
                self.target = nearestHome
                return

        self.target = None

    def chooseAttackAction(self, gameState):
        self.getTargetFood(gameState)
        self.getAttackTarget(gameState)

        if self.target:
            greedyAction = self.getGreedyAction(gameState, self.target)
            return greedyAction
        else:
            uctAction = self.getUCTAction(gameState)
            return uctAction

    def getDefenceTarget(self, gameState):
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

        if self.target is None and gameState.getAgentState(self.index).isPacman:
            self.target = self.getNearestHome(gameState, self.index)

        # record the food list. it will be compared to next round's food list to determine opponent's position
        self.foodLastRound = self.getFoodYouAreDefending(gameState).asList()

        """
    when there are only 5 food dots remaining. defender patrol around these food rather than  boundary line.
    """
        if self.target is None and len(self.getFoodYouAreDefending(gameState).asList()) <= 5:
            self.target = random.choice(self.getDefendingTarget(gameState))
        elif self.target is None:
            # random to choose a position around the boundary to patrol.
            choices = self.patrolDict.keys()
            if self.lastPatrol and len(choices) > 1:
                choices.remove(self.lastPatrol)
            self.target = random.choice(choices)
            self.lastPatrol = self.target

    def chooseDefenderAction(self, gameState):
        self.getDefenceTarget(gameState)

        if gameState.getAgentState(self.index).isPacman:
            myGhostDist, nearestGhost = self.getNearestGhost(gameState)
            if nearestGhost:
                return self.aStarSearch(gameState)

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
        # print self.getTeam(gameState)
        teammates = self.getTeam(gameState)
        teammates.remove(self.index)
        teammate = teammates[0]
        matePosition = gameState.getAgentState(teammate).getPosition()
        if not gameState.getAgentState(teammate).isPacman:
            self.getDefenceTarget(gameState)
            return self.chooseDefenderAction(gameState)
        else:
            self.getDefenceTarget(gameState)
            return self.aStarSearch(gameState)

    def chooseHelpAttackerAction(self, gameState):
        self.getTargetFood(gameState)
        self.getAttackTarget(gameState)
        return self.chooseAttackAction(gameState)

    # def BFSWithoutPosition(self, gameState, goal, position):
    #     queue = util.Queue()
    #     visited = []
    #     path = []
    #     queue.push((gameState, path))
    #
    #     while not queue.isEmpty():
    #         currState, path = queue.pop()
    #         visited.append(currState)
    #         if currState.getAgentState(self.index).getPosition() == goal:
    #             if len(path) == 0:
    #                 return Directions.STOP
    #             return path[0]
    #         if currState.getAgentState(self.index).getPosition() == position:
    #             continue
    #         actions = currState.getLegalActions(self.index)
    #         actions.remove(Directions.STOP)
    #         for action in actions:
    #             nextState = currState.generateSuccessor(self.index, action)
    #             if not nextState in visited:
    #                 visited.append(nextState)
    #                 queue.push((nextState, path + [action]))
    #     return Directions.STOP
    
    
    
    

    # -----------------------------------something needed for A*-------------------------#

    def aStarSearch(self, gameState):
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

            nextMoves = self.getSuccessors(currentPos,actionList,gameState)
            for nextNode in nextMoves:
                new_cost = cost_so_far[currentPos] + nextNode[2]
                if nextNode[0][0] not in cost_so_far:
                    cost_so_far[nextNode[0][0]] = new_cost
                    priorityQueue.push((nextNode[0][0], actionList + [nextNode[1]]),
                                       new_cost + self.heuristic(nextNode[0][0],gameState))
                elif new_cost < cost_so_far[nextNode[0][0]]:
                    priorityQueue.push((nextNode[0][0], actionList + [nextNode[1]]),
                                       new_cost + self.heuristic(nextNode[0][0],gameState))

        return path[0]

    def getGhosts(self,gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        return ghosts

    def getCostFunction(self,nextState,gameState):
        agentState = gameState.data.agentStates[self.index]
        isPacman = agentState.isPacman
        cost = 0
        invaders = self.getInvaders(gameState)
        ghosts = self.getGhosts(gameState)
        invaderPos = [a.getPosition() for a in invaders]
        ghostPos = [a.getPosition() for a in ghosts]
        teammates = self.getTeam(gameState)
        teammates.remove(self.index)
        teammate = teammates[0]
        matePosition = gameState.getAgentState(teammate).getPosition()

        # TODO - scared time
        if isPacman:
            if nextState in ghostPos:
                cost = 9999
            else:
                cost = 1
        else:
            if nextState in invaderPos:
                cost = 0
            else:
                cost = 1
        if nextState == matePosition:
            cost += 9999

        return cost

    def getSuccessors(self, state, actionList,gameState):
        successors = []
        position = state

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitWall = self.walls[nextx][nexty]

            if not hitWall:
                nextState = (nextx, nexty)
                cost = self.getCostFunction(nextState,gameState)
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

    def chooseAction(self, gameState):
        start = time.time()

        self.attacking = self.evaluateEvironment(gameState)
        if self.isAttacker != self.attacking:
            print self.index, self.isAttacker, self.attacking
        if self.attacking:
            # preparing for defence
            self.foodLastRound = self.getFoodYouAreDefending(gameState).asList()
            action = self.chooseAttackAction(gameState)
        else:
            action = self.chooseHelpDefenderAction(gameState)

        self.moves.append(action)
        if self.isReversing():
            print "REVERSING!!!"
            action = Directions.STOP

        print 'MCT attack agent %d: %.4f' % (self.index, time.time() - start)
        return action


class Denfender(FlexibleAgent):
    def registerInitialState(self, gameState):
        FlexibleAgent.registerInitialState(self, gameState)
        self.isAttacker = False
        self.attacking = False

    # self.CleanPatrolPostions(gameState.data.layout.height)
    # self.distFoodToPatrol(gameState)

    def chooseAction(self, gameState):
        start = time.time()
        self.attacking = self.evaluateEvironment(gameState)

        if self.isAttacker != self.attacking:
            print self.index, self.isAttacker, self.attacking
        if self.attacking:
            # preparing for attack
            self.foodLastRound = self.getFoodYouAreDefending(gameState).asList()
            action = self.chooseHelpAttackerAction(gameState)
            print 'MCT defend agent %d: %.4f' % (self.index, time.time() - start)
        else:
            action = self.chooseDefenderAction(gameState)

        self.moves.append(action)
        if self.isReversing():
            print "REVERSING!!!"
            action = Directions.STOP

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