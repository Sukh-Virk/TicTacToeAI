import copy
import random
import time
import sys
import math
from collections import namedtuple

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')


def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


# MonteCarlo Tree Search support

class MCTS:  # Monte Carlo Tree Search implementation
    class Node:
        def __init__(self, state, par=None):
            self.state = copy.deepcopy(state)

            self.parent = par
            self.children = []
            self.visitCount = 0
            self.winScore = 0

        def getChildWithMaxScore(self):
            maxScoreChild = max(self.children, key=lambda x: x.visitCount)
            return maxScoreChild

    def __init__(self, game, state):
        self.root = self.Node(state)
        self.state = state
        self.game = game
        self.exploreFactor = math.sqrt(2)

    def isTerminalState(self, utility, moves):
        return utility != 0 or len(moves) == 0

    def monteCarloPlayer(self, timelimit=4):

        start = time.perf_counter()
        end = start + timelimit
        maxd = 0


        while time.perf_counter() < end:

            currentd = 0

            currn = self.root

            while currn.children:
                currn = self.findBestNodeWithUCT(currn)
                currentd += 1

            maxd = max(maxd, currentd)

            if not self.isTerminalState(currn.state.utility, currn.state.moves):
                self.expandNode(currn)

            simulationResult = self.simulateRandomPlay(currn)
            self.backPropagation(currn, simulationResult)

        winnerNode = self.root.getChildWithMaxScore()

        assert (winnerNode is not None)

        print(f"Maximum depth reached: {maxd}")
        return (winnerNode.state.move)

    def selectNode(self, nd):

        if len(nd.children) == 0:
            return nd
        else:
            return self.selectNode(self.findBestNodeWithUCT(nd))

    def findBestNodeWithUCT(self, nd):

        bestn = None
        bestu = -float('inf')

        for child in nd.children:

            uct = self.uctValue(nd.visitCount, child.winScore, child.visitCount)
            if uct > bestu:
                bestn = child
                bestu = uct

        return bestn

    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        if nodeVisit == 0:
            return float('inf')

        avg = nodeScore / nodeVisit

        log_parent_visits = math.log(parentVisit)

        visit_ratio = log_parent_visits / nodeVisit

        exploration_term = self.exploreFactor * math.sqrt(visit_ratio)

        val = avg + exploration_term
        return val

    def expandNode(self, nd):

        stat = nd.state
        tempState = GameState(to_move=stat.to_move, move=stat.move, utility=stat.utility, board=stat.board,
                              moves=stat.moves)
        for a in self.game.actions(tempState):
            childNode = self.Node(self.game.result(tempState, a), nd)
            nd.children.append(childNode)

    def simulateRandomPlay(self, nd):



        wins = self.game.compute_utility(nd.state.board, nd.state.move, nd.state.board.get(nd.state.move, 0))

        if wins != 0:

            return 'X' if wins > 0 else 'O'

        s = copy.deepcopy(nd.state)

        for move in s.moves:

            ps = self.game.result(s, move)

            if self.game.compute_utility(ps.board, move, ps.board.get(move, 0)) > 0:
                if s.to_move == 'X':
                    return 'X'
                else:
                    return 'O'

        while not self.isTerminalState(s.utility, s.moves):

            rm = random.choice(s.moves)

            s = self.game.result(s, rm)


        finalUtility = self.game.compute_utility(s.board, s.move,
                                                 s.board.get(s.move, 0))
        return 'X' if finalUtility > 0 else 'O' if finalUtility < 0 else 'N'

    def backPropagation(self, nd, winningPlayer):
        """Backpropagation phase: Update score and visit count from the current leaf node to the root node."""
        tempNode = nd

        while tempNode:

            tempNode.visitCount = tempNode.visitCount +1

            if tempNode.state.to_move != winningPlayer:

                tempNode.winScore = tempNode.winScore +1

            tempNode = tempNode.parent




