# import necessary modules and classes
from util import manhattanDistance
from game import Directions, Agent
import random, util

# defining a class for the Reflex Agent
class ReflexAgent(Agent):
    # defining the action to be taken by the Reflex Agent
    def Action(self, gameState):
        # getting legal actions for the current game state
        legalActions = gameState.getLegalActions()
        # evaluating each action's successor and store the scores in a list
        scores = [self.evaluate(gameState.generatePacmanSuccessor(action)) for action in legalActions]
        # finding the highest score among the actions
        bestScore = max(scores)
        # selecting actions with the best score randomly if there are multiple
        bestActions = [action for action, score in zip(legalActions, scores) if score == bestScore]
        return random.choice(bestActions)

    # defining an evaluation function for the Reflex Agent
    def evaluate(self, successorGameState):
        # the evaluation function simply returns the game score of the successor state
        return successorGameState.getScore()

# defining an evaluation function for the game state's current score
def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()

# defining aclass for Adversarial Search Agent
class AdversarialSearchAgent(Agent):
    def __init__(self, evaluationFunction='scoreEvaluationFunction', depth=2):
        self.index = 0
        # setting the evaluation function based on the provided function name (default is scoreEvaluationFunction)
        self.evaluationFunction = util.lookup(evaluationFunction, globals())
        # setting the search depth for the agent
        self.depth = int(depth)

# defining a class for the Minimax Agent, which extends the Adversarial Search Agent
class MinimaxAgent(AdversarialSearchAgent):
    # checking if the current state is a terminal state based on the game rules and search depth
    def isTerminalState(self, state, depth):
        return state.isWin() or state.isLose() or depth == self.depth

    # implementing the Minimax algorithm to find the best action
    def minimax(self, state, depth, agent, maximize):
        # checking if the current state is terminal and return the evaluation if it is
        if self.isTerminalState(state, depth):
            return {"action": None, "score": self.evaluationFunction(state)}
        
        result = {"action": None, "score": float("inf") if agent else float("-inf")}
        
        # iterating over legal actions for the current agent
        for action in state.getLegalActions(agent):
            # generating a successor state by taking the action
            newState = state.generateSuccessor(agent, action)
            if maximize:
                # ff maximizing agent, recursively call minimax for the opponent (agent 1)
                score = self.minimax(newState, depth, 1, False)["score"]
                # updating the result if this action has a higher score
                result["action"], result["score"] = [(result["action"], result["score"]), (action, score)][score > result["score"]]
            else:
                if agent >= state.getNumAgents() - 1:
                    # if this agent is the last in the turn, increment the depth and switch to the next agent (agent 0)
                    score = self.minimax(newState, depth + 1, 0, True)["score"]
                else:
                    # otherwise continuing with the same agent for the next action
                    score = self.minimax(newState, depth, agent + 1, False)["score"]
                # updating the result with the minimum score
                result["score"] = min(score, result["score"])
        
        return result

    # defining the action to be taken by the Minimax Agent
    def Action(self, gameState):
        # calling the minimax function to find the best action and return it
        return self.minimax(gameState, 0, 0, True)["action"]

# defining a class for the Alpha-Beta Pruning Agent, which extends the Adversarial Search Agent
class AlphaBetaAgent(AdversarialSearchAgent):
    # checking if the current state is a terminal state based on the game rules and search depth
    def isTerminalState(self, state, depth):
        return state.isWin() or state.isLose() or depth == self.depth
    
    # implementing the Alpha-Beta Pruning algorithm to find the best action
    def alphaBeta(self, state, depth, agent, maximize, alpha, beta):
        # checking if the current state is terminal and return the evaluation if it is
        if self.isTerminalState(state, depth):
            return {"action": None, "score": self.evaluationFunction(state)}
        result = {"action": None, "score": float("inf") if agent else float("-inf")}
        # iterating over legal actions for the current agent
        for action in state.getLegalActions(agent):
            # generating a successor state by taking the action
            newState = state.generateSuccessor(agent, action)
            if maximize:
                # if maximizing agent, recursively call alphaBeta for the opponent (agent 1)
                score = self.alphaBeta(newState, depth, 1, False, alpha, beta)["score"]
                # updating the result if this action has a higher score
                result["action"], result["score"] = [(result["action"], result["score"]), (action, score)][score > result["score"]]
                # performing alpha-beta pruning
                if result["score"] >= beta:
                    break
                alpha = max(alpha, result["score"])
            else:
                if agent == state.getNumAgents() - 1:
                    # if this agent is the last in the turn, increment the depth and switch to the next agent (agent 0)
                    score = self.alphaBeta(newState, depth + 1, 0, True, alpha, beta)["score"]
                else:
                    # otherwise continuing with the same agent for the next action
                    score = self.alphaBeta(newState, depth, agent + 1, False, alpha, beta)["score"]
                # update the result with the minimum score
                result["score"] = min(score, result["score"])
                # performing alpha-beta pruning
                if alpha >= result["score"]:
                    break
                beta = min(beta, result["score"])
        return result
    # defining the action to be taken by the Alpha-Beta Pruning Agent
    def Action(self, gameState):
        # calling the alphaBeta function to find the best action and return it
        return self.alphaBeta(gameState, 0, 0, True, float("-inf"), float("inf"))["action"]

# defining a class for the Expectimax Agent, which extends the Adversarial Search Agent
class ExpectimaxAgent(AdversarialSearchAgent):
    # checking if the current state is a terminal state based on the game rules and search depth
    def isTerminalState(self, state, depth):
        return state.isWin() or state.isLose() or depth == self.depth

    # implementing the Expectimax algorithm to find the best action
    def expectimax(self, state, depth, agent, maximize):
        # checking if the current state is terminal and return the evaluation if it is
        if self.isTerminalState(state, depth):
            return {"action": None, "score": self.evaluationFunction(state)}
        
        result = {"action": None, "score": 0.0 if agent else float("-inf")}
        
        # defining a probability function for averaging expected scores
        prob = lambda x: x / len(state.getLegalActions(agent))
        
        # iterating over legal actions for the current agent
        for action in state.getLegalActions(agent):
            # generating a successor state by taking the action
            newState = state.generateSuccessor(agent, action)
            if maximize:
                # if maximizing agent, recursively call expectimax for the opponent (agent 1)
                score = self.expectimax(newState, depth, 1, False)["score"]
                # update the result if this action has a higher score
                result["action"], result["score"] = [(result["action"], result["score"]), (action, score)][score > result["score"]]
            else:
                if agent >= state.getNumAgents() - 1:
                    # if this agent is the last in the turn, increment the depth and switch to the next agent (agent 0)
                    score = self.expectimax(newState, depth + 1, 0, True)["score"]
                else:
                    # otherwise continue with the same agent for the next action
                    score = self.expectimax(newState, depth, agent + 1, False)["score"]
                # updating the result by adding the expected score (average)
                result["score"] += prob(score)
        
        return result
    
    # defining the action to be taken by the Expectimax Agent
    def Action(self, gameState):
        # calling the expectimax function to find the best action and return it
        return self.expectimax(gameState, 0, 0, True)["action"]
