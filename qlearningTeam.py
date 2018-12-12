# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
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

    def getInvaders(self, gameState):
        enemies = [gameState.getAgentState(each) for each in self.getOpponents(gameState)]
        return filter(lambda x: x.isPacman and x.getPosition() != None, enemies)

    def getDefendingTarget(self, gameState):
        return self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)


class OffensiveReflexAgent(RandomNumberTeamAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
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
        self.counter = 0
        self.epsilon = 0.9
        self.alpha = 0.9
        self.discount = 0.9

        self.start = gameState.getAgentPosition(self.index)
        self.target = None
        self.totalFoodList = self.getFood(gameState).asList()

        # recording dict
        self.weightO = util.Counter()

        self.weightO['successorScore'] = 1
        self.weightO['distanceToFood'] = 1

        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height

        #TODO - is mid point at the WALL? PatrolPostions!!!!
        self.midPointLeft = ((self.width / 2.0)-1, (self.height / 2.0))
        self.midPointRight = ((self.width / 2.0)+1, (self.height / 2.0))
        if gameState.isOnRedTeam(self.index):
            self.midPoint = self.midPointLeft
        else:
            self.midPoint = self.midPointRight

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.getQvalue(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    actionToReturn = action
                    bestDist = dist
        else:
            if util.flipCoin(self.epsilon) and len(bestActions) >0:
                actionToReturn = random.choice(bestActions)
            else:
                actionToReturn = random.choice(actions)

        # get reward
        reward = self.getReward(gameState,actionToReturn)
        successor = gameState.generateSuccessor(self.index, actionToReturn)

        # update the weight
        self.update(gameState,actionToReturn,successor,reward)

        # return the action
        return actionToReturn

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        currentState = gameState.getAgentPosition(self.index)
        successor = gameState.generateSuccessor(self.index, action)
        pos2 = successor.getAgentState(self.index).getPosition()
        x,y = pos2
        pos2 = (int(x),int(y))

        if pos2 != util.nearestPoint(pos2):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getQvalue(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        # get weights
        weights = self.getWeights()
        # comupute dot product or features and weights
        dotProduct = features * weights
        return dotProduct

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = 1 / len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = 1 / minDistance
        return features

    def getWeights(self):
        return self.weightO

    def computeValueFromQValues(self, gameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        actions = gameState.getLegalActions(self.index)
        values = [self.getQvalue(gameState, a) for a in actions]
        maxQValue = max(values)

        return maxQValue

    def getReward(self,gameState,actionToReturn):
        reward = 0
        currentState = gameState.getAgentPosition(self.index)
        successor = gameState.generateSuccessor(self.index, actionToReturn)
        pos2 = successor.getAgentState(self.index).getPosition()
        x, y = pos2
        pos2 = (int(x), int(y))
        '''
        Score
        '''
        if gameState.getScore() < successor.getScore():
            reward += 100

        '''
        Distance to Food
        '''
        foodList = self.getFood(gameState).asList()
        distanceToFood1 = abs(min([self.getMazeDistance(currentState, food) for food in foodList]))
        distanceToFood2 = abs(min([self.getMazeDistance(pos2, food) for food in foodList]))
        if distanceToFood1 > distanceToFood2:
            reward += 100
        elif distanceToFood1 < distanceToFood2:
            reward -= 100
        return reward

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # differece = reward + gamma*Q(s', a') - Q(s,a)
        difference = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQvalue(state, action)
        weights = self.getWeights()
        # if weight vector is empty, initialize it to zero
        if len(weights) == 0:
            weights[(state, action)] = 0
        features = self.getFeatures(state, action)
        # iterate over features and multiply them by the learning rate (alpha) and the difference
        for key in features.keys():
            weights[key] += features[key] * self.alpha * difference
        # sum the weights to their corresponding newly scaled features
        # print(weights)
        # update weights
        self.weightO = weights.copy()


class DefensiveReflexAgent(RandomNumberTeamAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

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
        self.counter = 0
        self.epsilon = 0.9
        self.alpha = 0.9
        self.discount = 0.9

        self.start = gameState.getAgentPosition(self.index)
        self.target = None
        self.totalFoodList = self.getFood(gameState).asList()

        # recording dict
        self.weights = util.Counter()

        self.weights['DistanceToMID'] = 40
        self.weights['Stop'] = -1.166
        self.weights['Invaders'] = -19.48
        self.weights['#ofDefendingFood'] = 1
        self.weights['DistanceToNINV'] = 0.264
        self.weights["IsPacman-D"] = 1

        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height

        #TODO - is mid point at the WALL? PatrolPostions!!!!
        self.midPointLeft = ((self.width / 2.0)-2, (self.height / 2.0))
        self.midPointRight = ((self.width / 2.0)+2, (self.height / 2.0))
        if gameState.isOnRedTeam(self.index):
            self.midPoint = self.midPointLeft
        else:
            self.midPoint = self.midPointRight

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.getQvalue(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    actionToReturn = action
                    bestDist = dist
        else:
            if util.flipCoin(self.epsilon) and len(bestActions) >0:
                actionToReturn = random.choice(bestActions)
            else:
                actionToReturn = random.choice(actions)

        # get reward
        reward = self.getReward(gameState,actionToReturn)
        successor = gameState.generateSuccessor(self.index, actionToReturn)

        # update the weight
        self.update(gameState,actionToReturn,successor,reward)

        # return the action
        return actionToReturn


    def getFeatures(self, gameState, action):
        features = util.Counter()

        currentState = gameState.getAgentPosition(self.index)
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        pos2 = successor.getAgentState(self.index).getPosition()
        x, y = pos2
        pos2 = (int(x), int(y))

        '''
        determine the mid point through the agent is on red team or not
        PatrolPostions!!!
        '''
        DtoMidPoint2 = self.getMazeDistance(pos2, self.midPoint)

        '''
        feature 1 : distance to boundary line
        '''
        if len(foodList) > 0 and DtoMidPoint2 <> 0:

            features["DistanceToMID"] = abs(100.0/ DtoMidPoint2)
            print ("F1:")
            print(features["DistanceToMID"])

        '''
        feature 2 : distance to nearest capsule
        '''
        # capsuleList = self.getCapsulesYouAreDefending(gameState)
        # DtoCaps= [self.getMazeDistance(pos2, caps) for caps in capsuleList]
        # if len(capsuleList) > 0 and len(DtoCaps) > 0:
        #
        #     minDistance = min(DtoCaps)
        #     if minDistance <>0:
        #         features["DistanceToNC"]= abs(1/minDistance)

        '''
        feature 3 : number of invaders
        '''
        invaders = self.getInvaders(gameState)
        features["Invaders"]= -len(invaders)
        print ("F2:")
        print(features["Invaders"])

        '''
        feature 4 : distance to invaders
        '''
        if len(invaders) > 0:
            dists = [self.getMazeDistance(pos2, a.getPosition()) for a in invaders]
            '''
            if not be scared
            '''
            if gameState.getAgentState(self.index).scaredTimer == 0 and min(dists)<>0:
                features["DistanceToNINV"]= 1 /min(dists)
                features["DistanceToMID"] = 0
            elif gameState.getAgentState(self.index).scaredTimer != 0 and min(dists)<>0:
                features["DistanceToNINV"] = - (1 / min(dists))
                features["DistanceToMID"] = 0
        print ("F4:")
        print(features["DistanceToNINV"])

        '''
        feature 5 : defendingFood
        '''
        defendingFoods = self.getDefendingTarget(gameState)
        if len(defendingFoods) > 0:
            features["#ofDefendingFood"]= 1 / len(defendingFoods)

        '''
        feature 6: STOP action
        '''
        if action == Directions.STOP: features['Stop'] = 1

        '''
        feature 7: isPacman?
        '''
        if successor.getAgentState(self.index).isPacman: features["IsPacman-D"] = 1

        return features

    def getWeights(self):
        return self.weights

    def getReward(self,gameState,actionToReturn):
        reward = 0
        currentState = gameState.getAgentPosition(self.index)
        successor = gameState.generateSuccessor(self.index, actionToReturn)
        pos2 = successor.getAgentState(self.index).getPosition()
        x, y = pos2
        pos2 = (int(x), int(y))
        invaders = self.getInvaders(gameState)
        invaders2 = self.getInvaders(successor)
        '''
        reward to F1
        '''
        distanceToMid1 = abs(self.getMazeDistance(currentState, self.midPoint))
        distanceToMid2 = abs(self.getMazeDistance(pos2,self.midPoint))
        if distanceToMid2 < distanceToMid1:
            reward += 10
        elif distanceToMid2 > distanceToMid1:
            reward -= 10

        '''
        reward to F3
        '''
        if len(invaders) > len(invaders2):
            reward += 1
        elif len(invaders) > len(invaders2):
            reward -= 1

        '''
        reward to F4
        '''
        if len(invaders) > 0:
            dist1= [self.getMazeDistance(currentState, a.getPosition()) for a in invaders]
            dist2 = [self.getMazeDistance(pos2, a.getPosition()) for a in invaders]
            if abs(min(dist2)) == 0:
                reward += 10
            if abs(min(dist2)) < abs(min(dist1)):
                reward += 5
            elif abs(min(dist2)) > abs(min(dist1)):
                reward -= 5

        '''
        reward to F5
        '''
        defendingFood1 = self.getDefendingTarget(gameState)
        defendingFood2 = self.getDefendingTarget(successor)

        if len(defendingFood1) < len(defendingFood2):
            reward -= 5

        '''
        reward to F6
        '''
        if actionToReturn == "Stop":
            reward -= 1

        '''
        reward to F7
        '''
        if successor.getAgentState(self.index).isPacman:
            reward -= 5

        return reward

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # differece = reward + gamma*Q(s', a') - Q(s,a)
        difference = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQvalue(state, action)
        print(difference)
        weights = self.getWeights()
        # if weight vector is empty, initialize it to zero
        if len(weights) == 0:
            weights[(state, action)] = 0
        features = self.getFeatures(state, action)
        # iterate over features and multiply them by the learning rate (alpha) and the difference
        for key in features.keys():
            weights[key] += features[key] * self.alpha * difference
        # sum the weights to their corresponding newly scaled features
        print(weights)
        # update weights
        self.weights = weights.copy()

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        currentState = gameState.getAgentPosition(self.index)
        successor = gameState.generateSuccessor(self.index, action)
        pos2 = successor.getAgentState(self.index).getPosition()
        x,y = pos2
        pos2 = (int(x),int(y))

        if pos2 != util.nearestPoint(pos2):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getQvalue(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        # get weights
        weights = self.getWeights()
        # comupute dot product or features and weights
        dotProduct = features * weights
        return dotProduct

    def computeValueFromQValues(self, gameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        actions = gameState.getLegalActions(self.index)
        values = [self.getQvalue(gameState, a) for a in actions]
        maxQValue = max(values)

        return maxQValue
        
        

    
    #------------new-only-defense----------------#
    
def createTeam(firstIndex, secondIndex, isRed,
               first='RandomNumberTeamAgent', second='OffensiveReflexAgent'):
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
        self.counter = 0
        self.epsilon = 0.9
        self.alpha = 0.9
        self.discount = 0.9


        self.walls = gameState.getWalls()
        self.start = gameState.getAgentPosition(self.index)
        self.target = None
        actions = ['Stop', 'North', 'West', 'South', 'East']
        self.totalFoodList = self.getFood(gameState).asList()

        # recording dict
        self.weights = util.Counter()

        self.weights['DistanceToMid'] = 0.03
        self.weights['DistanceToTarget'] = 0.1
        # self.weights['stop'] = - 9.89
        # self.weights['Invaders'] = -506
        # self.weights['#ofDefendingFood'] = 20
        # self.weights['SuccessorScore'] = 0.0
        # self.weights['DistanceToNC'] = 0.0
        # self.weights['DistanceToNINV'] = 520.63

        self.FoodLastRound = None

        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height

        # TODO - is mid point at the WALL? PatrolPostions!!!!
        self.midPointLeft = ((self.width / 2.0)-1, (self.height / 2.0))
        self.midPointRight = ((self.width / 2.0)+1, (self.height / 2.0))
        if gameState.isOnRedTeam(self.index):
            self.midPoint = self.midPointLeft
        else:
            self.midPoint = self.midPointRight




    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.getQvalue(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    actionToReturn = action
                    bestDist = dist
        else:
            if util.flipCoin(self.epsilon) and len(bestActions) >0:
                actionToReturn = random.choice(bestActions)
            else:
                actionToReturn = random.choice(actions)

        reward = self.getReward(gameState,actionToReturn)
        successor = gameState.generateSuccessor(self.index, actionToReturn)
        self.update(gameState,actionToReturn,successor,reward)
        return actionToReturn


    def getReward(self,gameState,actionToReturn):
        reward = 0
        currentState = gameState.getAgentPosition(self.index)
        x1,y1 = currentState
        successor = gameState.generateSuccessor(self.index, actionToReturn)
        pos2 = successor.getAgentState(self.index).getPosition()
        x, y = pos2
        pos2 = (int(x), int(y))
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        distanceToMid1 = abs(self.getMazeDistance(currentState, self.midPoint))
        distanceToMid2 = abs(self.getMazeDistance(pos2,self.midPoint))

        if len(invaders) == 0:
            if distanceToMid1 >= 5:
                if distanceToMid2 < distanceToMid1:
                    reward += 1
                elif distanceToMid2 > distanceToMid1:
                    reward -= 1


        if len(invaders) > 0:
            dist1=  min([self.getMazeDistance(currentState, a.getPosition()) for a in invaders])
            dist2 = min([self.getMazeDistance(pos2, a.getPosition()) for a in invaders])
            if dist2 ==0:
                reward += 3
            else:
                if abs(dist2) < abs(dist1):
                    reward += 1
                elif abs(dist2) > abs(dist1):
                    reward -= 1

                # if actionToReturn == "Stop":
                #     reward -= 1



        #print(reward)
        return reward

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        currentState = gameState.getAgentPosition(self.index)
        successor = gameState.generateSuccessor(self.index, action)
        pos2 = successor.getAgentState(self.index).getPosition()
        x,y = pos2
        pos2 = (int(x),int(y))

        if pos2 != util.nearestPoint(pos2):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getQvalue(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        # get weights
        weights = self.getWeights()
        # comupute dot product or features and weights
        dotProduct = features * weights
        return dotProduct

    def getFeatures(self, gameState, action):

        features = util.Counter()

        currentState = gameState.getAgentPosition(self.index)
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        pos2 = successor.getAgentState(self.index).getPosition()
        x, y = pos2
        pos2 = (int(x), int(y))


        # if action == Directions.STOP: features['stop'] = 1
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]



        '''
        feature 1 : distance to mid line
        '''

        if len(invaders) == 0:
            D2 = self.getMazeDistance(pos2, self.midPoint)
            if D2 <> 0:
                features["DistanceToMid"] = abs(1.0 / D2)
            else:
                features["DistanceToMid"] = 1.0

        # capsuleList = self.getCapsulesYouAreDefending(gameState)
        # DtoCaps= [self.getMazeDistance(pos2, caps) for caps in capsuleList]
        # if len(capsuleList) > 0 and len(DtoCaps) > 0:
        #     '''
        #     feature 2 : distance to nearest capsule
        #     '''
        #     minDistance = min(DtoCaps)
        #     if minDistance <>0:
        #         features["DistanceToNC"]= abs(1/minDistance)
        # else:
        #     features["DistanceToNC"] = 0


        # defendingFood = self.getFoodYouAreDefending(gameState).asList()
        # '''
        # feature 3 : number of invaders
        # '''
        # features["Invaders"]= len(invaders)

        if len(invaders) > 0:
            dist = min([self.getMazeDistance(currentState, a.getPosition()) for a in invaders])

            '''
            if not be scared
            '''
            if gameState.getAgentState(self.index).scaredTimer == 0 and dist <>0:
                features["DistanceToTarget"]= abs(1 / dist)
            elif gameState.getAgentState(self.index).scaredTimer != 0 and dist <>0:
                features["DistanceToTarget"] = - abs(1 / dist)
            elif  gameState.getAgentState(self.index).scaredTimer == 0 and dist == 0:
                features["DistanceToTarget"] = 1
            elif gameState.getAgentState(self.index).scaredTimer != 0 and dist == 0:
                features["DistanceToTarget"] = - 1



        '''
        # feature 5 : defendingFood
        # '''
        # if len(defendingFood) > 0:
        #     features["#ofDefendingFood"]= len(defendingFood) / len(self.totalFoodList)
        # else:
        #     features["#ofDefendingFood"] = 0

        return features


    def getWeights(self):
        return self.weights


    def computeValueFromQValues(self, gameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        actions = gameState.getLegalActions(self.index)
        values = [self.getQvalue(gameState, a) for a in actions]
        maxQValue = max(values)

        return maxQValue


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # differece = reward + gamma*Q(s', a') - Q(s,a)
        featureVector = self.getFeatures(state, action)

        maxQFromNextState = self.computeValueFromQValues(nextState)
        actionQValue = self.getQvalue(state, action)

        correction = (reward + self.discount * maxQFromNextState - actionQValue)
        print maxQFromNextState - actionQValue
        #print(correction)


        for feature in featureVector:
            self.weights[feature] += self.alpha * correction* featureVector[feature]

        print(self.weights)



