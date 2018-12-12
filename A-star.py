from captureAgents import CaptureAgent

import random, time, util
import capture as cp

from game import Directions
from game import Agent
from game import Actions
import game


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class RandomNumberTeamAgent(CaptureAgent):
    counter = None  # type: int

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

        # set variables
        self.count = 0
        self.lastCount = 0
        self.patrolPositions = self.getPatrolPosition(gameState)

        self.walls = gameState.getWalls()
        self.start = gameState.getAgentPosition(self.index)
        self.target = None
        actions = ['Stop', 'North', 'West', 'South', 'East']
        self.totalFoodList = self.getFood(gameState).asList()

        self.FoodLastRound = None

        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        print self.height

        self.initialDefendingPos = self.patrolPositions[int(len(self.patrolPositions)/2)]
        #print(self.initialDefendingPos)

        self.lastGoal1 = None
        self.lastGoal2 = None

        # action plans
        self.path = []
        self.lastState = None
        self.lastAction = None
        self.posNum = 0

        self.lastFoodList = self.getFood(gameState).asList()

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        if gameState.isOnRedTeam(self.index) and "East" in actions:
            actions.remove("East")
        elif not gameState.isOnRedTeam(self.index) and "West" in actions:
            actions.remove("West")

        start = time.time()

        path = self.getActionPlans(gameState)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()

        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        if len(path) > 0:
            action = path[0]
            path.remove(action)
        else:
            action = random.choice(actions)

        self.lastState = gameState
        print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        return action

    def getActionPlans(self, gameState):
        self.path = self.aStarSearch(gameState)

        return self.path

    def getSuccessors(self, state, actionList, gameState):
        successors = []
        position = state

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitWall = self.walls[nextx][nexty]

            if not hitWall:
                nextState = (nextx, nexty)
                cost = self.getCostFunction(nextState, gameState)
                successors.append(((nextState, actionList), action, cost))
        return successors

    def getCostFunction(self, nextState, gameState):
        agentState = gameState.data.agentStates[self.index]
        isPacman = agentState.isPacman
        cost = 0
        invaders = self.getInvaders(gameState)
        ghosts = self.getGhosts(gameState)
        invaderPos = [a.getPosition() for a in invaders]
        ghostPos = [a.getPosition() for a in ghosts]
        scaredGhostPos = []

        for index in self.getOpponents(gameState):
            if gameState.getAgentState(index).scaredTimer != 0:
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
            if gameState.getAgentState(self.index).scaredTimer == 0:
                if nextState in invaderPos:
                    cost = 0
                else:
                    cost = 1
            else:
                if nextState in invaderPos:
                    cost = 9999
                else:
                    cost = 1

            nextState_x, nextState_y = nextState
            if gameState.isOnRedTeam(self.index):
                midWidth = int((self.width)/2)
                if nextState_x >= midWidth:
                    cost += 9999
            else:
                midWidth = int((self.width)/2)+1
                if nextState_x <= midWidth:
                    cost += 9999


        return cost

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos2 = successor.getAgentState(self.index).getPosition()
        x, y = pos2
        pos2 = (int(x), int(y))

        if pos2 != util.nearestPoint(pos2):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # --------------------------Goal Setting------------------------#

    def isGoalState(self, gameState, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        currentPos = state
        goalPosition = self.getGoalPosition(gameState)

        if currentPos == goalPosition:
            isGoal = True
        else:
            isGoal = False

        return isGoal

    def getCurrentPos(self, gameState):
        currentPos = gameState.getAgentPosition(self.index)
        return currentPos

    def getGoalPosition(self, gameState):
        goalPosition = None
        return goalPosition

    # -----------------Getting Information From layout-------------------#

    def getInvaders(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        return invaders

    def getGhosts(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        return ghosts

    def getDefendingTarget(self, gameState):
        return self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)

    def defendingTargetChanges(self, gameState):
        if self.lastState == None:
            return False
        else:
            return len(self.getDefendingTarget(self.lastState)) == len(self.getDefendingTarget(gameState))

    def nearestGhostInfo(self, gameState):
        currentPos = self.getCurrentPos(gameState)
        invaders = self.getInvaders(gameState)
        minDistance = 9999
        if len(invaders) > 0:
            for a in invaders:
                distance = self.manhattonDistance(currentPos, a.getPosition())
                if distance <= minDistance:
                    minDistance = distance
                    pos = a.getPosition()
                    return pos, minDistance
        return None, -9999

    def getNearestHome(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        nearestHomeDist, nearestHome = min([(self.getMazeDistance(returnPos, myPos), returnPos)
                                            for returnPos in self.patrolPositions])
        return nearestHome

    def getPatrolPosition(self, gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        patrolPositions = []
        if self.red:
            centralX = (width - 2) / 2
        else:
            centralX = ((width - 2) / 2) + 1
        for i in range(1, height - 1):
            if not gameState.hasWall(centralX, i):
                patrolPositions.append((centralX, i))
        return patrolPositions


    def heuristic(self, state, gameState):

        h = {}
        hValue = 0

        goalPosition = self.getGoalPosition(gameState)

        if self.manhattonDistance(state, goalPosition) <= 6:
            hValue = self.getMazeDistance(state, goalPosition)
        else:
            hValue = self.manhattonDistance(state, goalPosition)

        return hValue

    def aStarSearch(self, gameState):
        """Search the node that has the lowest combined cost and heuristic first."""
        currentPosition = self.getCurrentPos(gameState)
        path = []

        currentPos = currentPosition
        priorityQueue = util.PriorityQueue()
        cost_so_far = {}
        priorityQueue.push((currentPos, []), 0)
        cost_so_far[currentPos] = 0

        while not priorityQueue.isEmpty():
            currentPos, actionList = priorityQueue.pop()

            if self.isGoalState(gameState, currentPos):
                path = actionList

            nextMoves = self.getSuccessors(currentPos, actionList, gameState)
            for nextNode in nextMoves:
                new_cost = cost_so_far[currentPos] + nextNode[2]
                if nextNode[0][0] not in cost_so_far:
                    cost_so_far[nextNode[0][0]] = new_cost
                    priorityQueue.push((nextNode[0][0], actionList + [nextNode[1]]),
                                       new_cost + self.heuristic(nextNode[0][0], gameState))
                elif new_cost < cost_so_far[nextNode[0][0]]:
                    priorityQueue.push((nextNode[0][0], actionList + [nextNode[1]]),
                                       new_cost + self.heuristic(nextNode[0][0], gameState))

        return path

    def manhattonDistance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        distance = abs(x2 - x1) + abs(y2 - y1)
        return distance

    def getNearestFood(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        foods = self.getFood(gameState).asList()
        oneStepFood = len([food for food in foods if self.getMazeDistance(myPos, food) == 1])
        twoStepFood = len([food for food in foods if self.getMazeDistance(myPos, food) == 2])
        passBy = (oneStepFood == 1 and twoStepFood == 0)
        if len(foods) > 0:
            return min([(self.getMazeDistance(myPos, food), food, passBy) for food in foods])
        return None, None, None

    def getNearestCapsule(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            return min([(self.getMazeDistance(myPos, cap), cap) for cap in capsules])
        return -9999, None

    def updateDefendingPos(self, gameState):
        self.partrolPosition = self.getPatrolPosition(gameState)
        self.lenPP = len(self.patrolPositions)
        #print self.partrolPosition
        positionNumber = self.posNum % self.lenPP
        self.initialDefendingPos = self.getPatrolPosition(gameState)[positionNumber]
        #print self.initialDefendingPos
        self.posNum = self.posNum + 1




class OffensiveAgent(RandomNumberTeamAgent):

    def getGoalPosition(self, gameState):
        invaders = self.getInvaders(gameState)
        ghosts = self.getGhosts(gameState)
        ghostPos, distanceToGhost = self.nearestGhostInfo(gameState)
        foods = self.getFood(gameState).asList()
        distanceTofood, foodPos, passBy = self.getNearestFood(gameState)
        Agent = gameState.data.agentStates[self.index]
        dToCapsule, capsulePos = self.getNearestCapsule(gameState)

        if len(ghosts) == 0:
            if len(foods) > 0 and Agent.numCarrying <= (len(foods) / 9.0):
                goalPosition = foodPos
            else:
                goalPosition = self.getNearestHome(gameState)
        else:
            if self.ghostComing(gameState):
                if dToCapsule < (distanceToGhost / 2.0) and dToCapsule != -9999 and distanceToGhost != 9999:
                    goalPosition = capsulePos
                elif distanceToGhost >= 4:
                    goalPosition = foodPos
                else:
                    goalPosition = self.getNearestHome(gameState)
            else:
                if len(foods) > 0 and Agent.numCarrying <= (len(foods) / 9.0):
                    goalPosition = foodPos
                else:
                    goalPosition = self.getNearestHome(gameState)

        return goalPosition

    def ghostComing(self, gameState):
        for index in self.getOpponents(gameState):
            if gameState.getAgentState(index).scaredTimer == 0:
                return True
        return False

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        if gameState.isOnRedTeam(self.index) and "East" in actions:
            actions.remove("East")
        elif not gameState.isOnRedTeam(self.index) and "West" in actions:
            actions.remove("West")

        path = self.getActionPlans(gameState)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()

        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        if len(path) > 0:
            action = path[0]
            path.remove(action)
        else:
            action = random.choice(actions)

        self.lastState = gameState
        return action



class DefensiveAgent(RandomNumberTeamAgent):

    def getGoalPosition(self, gameState):
        invaders = self.getInvaders(gameState)
        ghostPos, distanceTo = self.nearestGhostInfo(gameState)
        beScared = gameState.getAgentState(self.index).scaredTimer != 0

        if len(invaders) == 0:
            goalPosition = self.initialDefendingPos

        if len(invaders) > 0:

            if not beScared:
                goalPosition = ghostPos
            else:
                goalPosition = self.initialDefendingPos

        #print(goalPosition)
        return goalPosition

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        actions.remove("Stop")

        if gameState.isOnRedTeam(self.index) and "East" in actions:
            actions.remove("East")
        elif not gameState.isOnRedTeam(self.index) and "West" in actions:
            actions.remove("West")


        path = self.getActionPlans(gameState)
        if len(path) > 0:
            action = path[0]
            path.remove(action)
        else:
            self.updateDefendingPos(gameState)
            action = random.choice(actions)

        self.lastState = gameState
        return action
