# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foods = currentGameState.getFood().asList()
        distance = []

        for ghost in newGhostStates:
            if ghost.getPosition() == newPos:
                return -float("inf")
        for food in foods:
            distance.append(-1*(util.manhattanDistance(newPos, food)))
        return max(distance)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides so me common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxFunction(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            maxVal = -float("inf")

            actions = gameState.getLegalActions(agentIndex = 0)
            for action in actions:
                maxVal = max(maxVal, minFunction(gameState.generateSuccessor(0, action),depth, 1))

            return maxVal

        def minFunction(gameState, depth, ghost):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            
            minVal = float("inf")
            actions = gameState.getLegalActions(agentIndex = ghost)
            if ghost != gameState.getNumAgents()-1:                
                for action in actions:
                    minVal = min(minVal, minFunction(gameState.generateSuccessor(ghost, action), depth, ghost+1))
            else:                    
                for action in actions:
                    minVal = min(minVal, maxFunction(gameState.generateSuccessor(gameState.getNumAgents()-1, action), depth-1))

            return minVal

        actions = gameState.getLegalActions()
        best = Directions.STOP

        score = -float("inf")
        maxScore = -float("inf")
        for action in actions:
            maxScore = score
            score = max(score, minFunction(gameState.generateSuccessor(0, action), self.depth, 1))

            if score > maxScore:
                maxScore = score
                best = action

        return best
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxFunction(gameState, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            maxVal = -float("inf")

            actions = gameState.getLegalActions(agentIndex = 0)
            for action in actions:
                maxVal = max(maxVal, minFunction(gameState.generateSuccessor(0, action),depth, 1, alpha, beta))
                if maxVal > beta:
                    return maxVal
                alpha = max(alpha, maxVal)
            return maxVal

        def minFunction(gameState, depth, ghost, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            
            minVal = float("inf")
            actions = gameState.getLegalActions(agentIndex = ghost)
            if ghost != gameState.getNumAgents()-1:                
                for action in actions:
                    minVal = min(minVal, minFunction(gameState.generateSuccessor(ghost, action), depth, ghost+1, alpha, beta))
                    if minVal < alpha:
                        return minVal
                    beta = min(beta, minVal)
            else:                    
                for action in actions:
                    minVal = min(minVal, maxFunction(gameState.generateSuccessor(gameState.getNumAgents()-1, action), depth-1, alpha, beta))
                    if minVal < alpha:
                        return minVal
                    beta = min(beta, minVal)

            return minVal

        actions = gameState.getLegalActions()
        best = Directions.STOP

        score = -float("inf")
        maxScore = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        for action in actions:
            maxScore = score
            score = max(score, minFunction(gameState.generateSuccessor(0, action), self.depth, 1, alpha, beta))

            if score > maxScore:
                maxScore = score
                best = action
            if score > beta:
                return best

            alpha = max(alpha, maxScore)
        return best
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maxFunction(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            maxVal = -float("inf")

            actions = gameState.getLegalActions(agentIndex = 0)
            for action in actions:
                maxVal = max(maxVal, expectFunction(gameState.generateSuccessor(0, action),depth, 1))

            return maxVal

        def expectFunction(gameState, depth, ghost):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex = ghost)
            expected = 0
            numOfActions = len(actions)

            if ghost != gameState.getNumAgents()-1:                
                for action in actions:
                    expected +=  expectFunction(gameState.generateSuccessor(ghost, action), depth, ghost+1)
            else:                    
                for action in actions:
                    expected += maxFunction(gameState.generateSuccessor(gameState.getNumAgents()-1, action), depth-1)

            chanceVal = expected/numOfActions
            return chanceVal

      
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex = 0)        
        best = Directions.STOP
        score = -float("inf")

        for action in actions:
            maxScore = score
            score = max(score, expectFunction(gameState.generateSuccessor(0, action), self.depth, 1))

            if score > maxScore:
                maxScore = score
                best = action

        return best
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    distance to food pellet + distance to capsule _+ 
    """
    "*** YOUR CODE HERE ***"

    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    foodsList = foods.asList()    
    caps = currentGameState.getCapsules()

    minDist = -1
    for food in foodsList:
        minDist = min(minDist, util.manhattanDistance(pos, food))
        if minDist == -1:
            minDist =  util.manhattanDistance(pos, food)

    ghostDist = 1
    for ghost in currentGameState.getGhostPositions():
        ghostDist += util.manhattanDistance(pos, ghost)
    
    return currentGameState.getScore() + (1 / minDist) - (1 / ghostDist) - len(caps)

    util.raiseNotDefined()

def arcConsistencyCSP(csp):
    """
    Implement AC3 here
    """
    "*** YOUR CODE HERE ***"
    #print(csp.getDomains())
    #print(csp.getVars())
    #print(csp.getConstraintGraph())
    def revise(Xi, Xj, domains):
        revised = False
        Di = domains[Xi]
        Dj = domains[Xj]
        f = 1
        b = ''
        if len(Dj) == 1:
            for i,x in enumerate(Di):
                for y in Dj:
                    if y == x and f:
                        f = 0
                        b = x
                  
        if not f:#condition
            Di.remove(b)
            domains[Xi] = Di
            revised = True

        return(revised, domains)

    queue = list(csp.getConstraintGraph())
    graph = list(csp.getConstraintGraph())
    domains = csp.getDomains()

    for edges in graph:
        queue.append([edges[1], edges[0]])

    graph = queue.copy()
        
    while queue:
        Xi,Xj = queue.pop(0)
        revised, domains = revise(Xi, Xj, domains)
        
        if revised:
            if len(domains[Xi]) == 0:
                return {}    

            for n in graph:
                if n[1] == Xi:
                    if n not in queue:
                        queue.append(n)
    return domains

# Abbreviation
better = betterEvaluationFunction
