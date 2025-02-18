"""Games or Adversarial Search (Chapter 5)"""

import copy
import random
from collections import namedtuple
import numpy as np
import time

# namedtuple used to generate game state:
GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')


def gen_state(move='(1, 1)', to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """
        move = the move that has lead to this state,
        to_move=Whose turn is to move
        x_position=positions on board occupied by X player,
        o_position=positions on board occupied by O player,
        (optionally) number of rows, columns and how many consecutive X's or O's required to win,
    """
    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, move=move, utility=0, board=board, moves=moves)


# ______________________________________________________________________________
# MinMax Search
def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states (full depth search)."""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for action in game.actions(state):
            v = max(v, min_value(game.result(state, action)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for action in game.actions(state):
            v = min(v, max_value(game.result(state, action)))
        return v


    best_score = -np.inf

    best_action = None

    actions = game.actions(state)

    index = 0

    while index < len(actions):

        action = actions[index]

        value = min_value(game.result(state, action))
        if value > best_score:

            best_score = value

            best_action = action

        index = index +1

    return best_action



def evaldetection(game, state):

    player = state.to_move
    opponent = 'O' if player == 'X' else 'X'
    board = state.board
    k = game.k
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    index = 0
    while index < len(state.moves):

        pos = state.moves[index]
        oth = 0
        while oth < len(directions):

            direction = directions[oth]

            if game.k_in_row(board, pos, opponent, direction, k - 1):

                return -10000
            oth = oth +1
        index = index +1
    return 0



def minmax_cutoff(game, state):

    depthl = 3 if len(state.moves) > 10 else 5
    player = game.to_move(state)

    def max_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player) * 1000
        if depth == 0:
            threat = evaldetection(game, state)
            if threat != 0:
                return threat
            else:
                return game.eval1(state)
        best = -np.inf
        for action in game.actions(state):
            vals = min_value(game.result(state, action), depth-1)
            best = max(best, vals)
        return best

    def min_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player) * 1000
        if depth == 0:
            threat = evaldetection(game, state)
            if not threat == 0:
                return threat
            else:
                return game.eval1(state)
        best = np.inf
        for action in game.actions(state):
            vals = max_value(game.result(state,action), depth-1)
            best = min(best, vals)
        return best

    best_score = -np.inf
    best_action = None

    for action in game.actions(state):
        value = min_value(game.result(state, action), depthl - 1)
        if value > best_score:
            best_score = value
            best_action = action

    return best_action




# ______________________________________________________________________________
def alpha_beta(game, state):

    player = game.to_move(state)

    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)

        vals = -np.inf
        actions = game.actions(state)
        index = 0

        while index < len(actions):

            action = actions[index]

            m = min_value(game.result(state, action), alpha, beta)
            vals = max(vals, m)

            if vals >= beta:

                return vals

            alpha = max(alpha, vals)

            index = index +1

        return vals

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        vals = np.inf
        for action in game.actions(state):
            m = max_value(game.result(state, action), alpha, beta)
            vals = min(vals, m)
            if vals <= alpha:
                return vals
            beta = min(beta, vals)
        return vals

    bests = -np.inf
    alp = -np.inf
    bet= np.inf
    besta = None


    actions = game.actions(state)
    index = 0

    while index < len(actions):
        act = actions[index]
        value = min_value(game.result(state, act), alp, bet)
        if value > bests:
            bests = value
            besta = act
            alpha = max(alp, bests)
    ind= index +1

    return besta


def alpha_beta_cutoff(game, state, depth=4):

    player = game.to_move(state)

    def max_value(state, alpha, beta, depth):
        if game.terminal_test(state):
            return game.utility(state, player) * 1000
        if depth == 0:
            threat= evaldetection(game, state)
            return threat if threat != 0 else game.eval1(state)

        vals = -np.inf
        actions = game.actions(state)
        loc = 0

        while loc < len(actions):

            act = actions[loc]

            m =  min_value(game.result(state, act), alpha, beta, depth - 1)
            vals = max(vals, m)
            if vals >= beta:
                return vals
            alpha = max(alpha, vals)
            loc = loc +1

        return vals

    def min_value(state, alpha, beta, depth):

        if game.terminal_test(state):

            return game.utility(state, player) * 1000
        if depth == 0:
            threat = evaldetection(game, state)

            return threat if threat != 0 else game.eval1(state)

        vals = np.inf
        for action in game.actions(state):
            m = max_value(game.result(state, action), alpha, beta, depth - 1)
            vals = min(vals, m)

            if vals < alpha or vals == alpha:
                return vals
            beta = min(beta, vals)
        return vals

    alp = -np.inf
    bet = np.inf
    bests = -np.inf
    besta = None

    for action in game.actions(state):
        value = min_value(game.result(state, action), alp, bet, depth - 1)


        if evaldetection(game, game.result(state, action)) == -10000:

            return action

        if value > bests:
            bests = value
            besta = action
        alp = max(alp, bests)

    return besta


def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    if game.timer < 0:
        game.d = -1
        return alpha_beta(game, state)

    tcells = game.size * game.size
    ecells = len(state.moves)

    # Use a random move if most cells are already filled
    if ecells >= tcells * 0.8:
        return random_player(game, state)

    beg = time.perf_counter()
    end = beg + game.timer
    pos = None
    depth = 1

    for depth in range(1, game.maxDepth + 1):
        if time.perf_counter() >= end:
            break
        bm = alpha_beta_cutoff(game, state, depth)
        pos = bm

    game.d = depth
    print("alpha_beta_player: iterative deepening to depth:", game.d)


    return pos if pos is not None else random.choice(state.moves)



def minmax_player(game, state):
    if (game.timer < 0):
        game.d = -1
        return minmax(game, state)

    start = time.perf_counter()

    if game.timer > 0:
        end = start + game.timer
    else:
        end = float('inf')

    pos= None
    depth = 1

    while time.perf_counter() < end:
        pos = minmax_cutoff(game, state)
        depth = depth +1
        if depth > game.maxDepth:
            break

    print(f"minmax_player: Iterative deepening reached depth {depth - 1}")
    return pos


# ______________________________________________________________________________
# base class for Games

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))


class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, size=3, k=3, t=-1):
        self.size = size
        if k <= 0:
            self.k = size
        else:
            self.k = k
        self.d = -1  # d is cutoff depth. Default is -1 meaning no depth limit. It is controlled usually by timer
        self.maxDepth = size * size  # max depth possible is width X height of the board
        self.timer = t  # timer  in seconds for opponent's search time limit. -1 means unlimited
        moves = [(x, y) for x in range(1, size + 1)
                 for y in range(1, size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def reset(self):
        moves = [(x, y) for x in range(1, self.size + 1)
                 for y in range(1, self.size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    @staticmethod
    def switchPlayer(player):
        assert (player == 'X' or player == 'O')
        return 'O' if player == 'X' else 'X'

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        try:
            moves = list(state.moves)
            moves.remove(move)
        except (ValueError, IndexError, TypeError) as e:
            print("exception: ", e)

        return GameState(to_move=self.switchPlayer(state.to_move), move=move,
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or lost or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(0, self.size):
            for y in range(1, self.size + 1):
                print(board.get((self.size - x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If player wins with this move, return k if player is 'X' and -k if 'O' else return 0."""
        if (self.k_in_row(board, move, player, (0, 1), self.k) or
                self.k_in_row(board, move, player, (1, 0), self.k) or
                self.k_in_row(board, move, player, (1, -1), self.k) or
                self.k_in_row(board, move, player, (1, 1), self.k)):
            return self.k if player == 'X' else -self.k
        else:
            return 0

    """Games or Adversarial Search (Chapter 5)"""

    import copy
    import random
    from collections import namedtuple
    import numpy as np
    import time

    # namedtuple used to generate game state:
    GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

    def gen_state(move='(1, 1)', to_move='X', x_positions=[], o_positions=[], h=3, v=3):
        """
            move = the move that has lead to this state,
            to_move=Whose turn is to move
            x_position=positions on board occupied by X player,
            o_position=positions on board occupied by O player,
            (optionally) number of rows, columns and how many consecutive X's or O's required to win,
        """
        moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
        moves = list(moves)
        board = {}
        for pos in x_positions:
            board[pos] = 'X'
        for pos in o_positions:
            board[pos] = 'O'
        return GameState(to_move=to_move, move=move, utility=0, board=board, moves=moves)


    def random_player(game, state):
        """A random player that chooses a legal move at random."""
        return random.choice(game.actions(state)) if game.actions(state) else None

    def alpha_beta_player(game, state):
        if game.timer < 0:
            game.d = -1
            return alpha_beta(game, state)

        tcells = game.size * game.size
        ecells = len(state.moves)

        if ecells >= tcells * 0.8:
            return random_player(game, state)

        start = time.perf_counter()
        end = start + game.timer
        move = None
        depth = 1  # Start with shallow depth and increase gradually

        while time.perf_counter() < end:
            best_move = alpha_beta_cutoff(game, state)
            if time.perf_counter() < end:
                move = best_move  # Keep track of the best move at the last completed depth
                depth = depth + 1
            else:
                break

        game.d = depth - 1
        print("alpha_beta_player: iterative deepening to depth:", game.d)

        return move if move else random.choice(state.moves)

    def minmax_player(game, state):
        if (game.timer < 0):
            game.d = -1
            return minmax(game, state)

        tcells = game.size * game.size
        ecells = len(state.moves)

        if ecells >= tcells * 0.8:
            return random_player(game, state)

        start = time.perf_counter()
        end = start + game.timer if game.timer > 0 else float('inf')

        move = None
        depth = 1

        while time.perf_counter() < end:
            move = minmax_cutoff(game, state, depth)
            depth += 1
            if depth > game.maxDepth:  # Prevent searching beyond the maximum possible depth
                break

        print(f"minmax_player: Iterative deepening reached depth {depth - 1}")
        return move

    # ______________________________________________________________________________
    # base class for Games

    class Game:
        """A game is similar to a problem, but it has a utility for each
        state and a terminal test instead of a path cost and a goal
        test. To create a game, subclass this class and implement actions,
        result, utility, and terminal_test. You may override display and
        successors or you can inherit their default methods. You will also
        need to set the .initial attribute to the initial state; this can
        be done in the constructor."""

        def actions(self, state):
            """Return a list of the allowable moves at this point."""
            raise NotImplementedError

        def result(self, state, move):
            """Return the state that results from making a move from a state."""
            raise NotImplementedError

        def utility(self, state, player):
            """Return the value of this final state to player."""
            raise NotImplementedError

        def terminal_test(self, state):
            """Return True if this is a final state for the game."""
            return not self.actions(state)

        def to_move(self, state):
            """Return the player whose move it is in this state."""
            return state.to_move

        def display(self, state):
            """Print or otherwise display the state."""
            print(state)

        def __repr__(self):
            return '<{}>'.format(self.__class__.__name__)

        def play_game(self, *players):
            """Play an n-person, move-alternating game."""
            state = self.initial
            while True:
                for player in players:
                    move = player(self, state)
                    state = self.result(state, move)
                    if self.terminal_test(state):
                        self.display(state)
                        return self.utility(state, self.to_move(self.initial))

    class TicTacToe(Game):
        """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
        A state has the player to_move, a cached utility, a list of moves in
        the form of a list of (x, y) positions, and a board, in the form of
        a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
        depth = -1 means max search tree depth to be used."""

        def __init__(self, size=3, k=3, t=-1):
            self.size = size
            if k <= 0:
                self.k = size
            else:
                self.k = k
            self.d = -1  # d is cutoff depth. Default is -1 meaning no depth limit. It is controlled usually by timer
            self.maxDepth = size * size  # max depth possible is width X height of the board
            self.timer = t  # timer  in seconds for opponent's search time limit. -1 means unlimited
            moves = [(x, y) for x in range(1, size + 1)
                     for y in range(1, size + 1)]
            self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

        def reset(self):
            moves = [(x, y) for x in range(1, self.size + 1)
                     for y in range(1, self.size + 1)]
            self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

        def actions(self, state):
            """Legal moves are any square not yet taken."""
            return state.moves

        @staticmethod
        def switchPlayer(player):
            assert (player == 'X' or player == 'O')
            return 'O' if player == 'X' else 'X'

        def result(self, state, move):
            if move not in state.moves:
                return state  # Illegal move has no effect
            board = state.board.copy()
            board[move] = state.to_move
            try:
                moves = list(state.moves)
                moves.remove(move)
            except (ValueError, IndexError, TypeError) as e:
                print("exception: ", e)

            return GameState(to_move=self.switchPlayer(state.to_move), move=move,
                             utility=self.compute_utility(board, move, state.to_move),
                             board=board, moves=moves)

        def utility(self, state, player):
            """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
            return state.utility if player == 'X' else -state.utility

        def terminal_test(self, state):
            """A state is terminal if it is won or lost or there are no empty squares."""
            return state.utility != 0 or len(state.moves) == 0

        def display(self, state):
            board = state.board
            for x in range(0, self.size):
                for y in range(1, self.size + 1):
                    print(board.get((self.size - x, y), '.'), end=' ')
                print()

        def compute_utility(self, board, move, player):
            """If player wins with this move, return k if player is 'X' and -k if 'O' else return 0."""
            if (self.k_in_row(board, move, player, (0, 1), self.k) or
                    self.k_in_row(board, move, player, (1, 0), self.k) or
                    self.k_in_row(board, move, player, (1, -1), self.k) or
                    self.k_in_row(board, move, player, (1, 1), self.k)):
                return self.k if player == 'X' else -self.k
            else:
                return 0



    def eval1(self, state):
        player = state.to_move
        opponent = 'O' if player == 'X' else 'X'
        board = state.board
        k = self.k
        threatl = k - 1


        if self.terminal_test(state):
            return self.utility(state, player) * 1000

        playersc = 0
        opponentt = 0
        centerb = 0

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]


        for pos in state.moves:
            for dir in directions:

                if self.k_in_row(board, pos, opponent, dir, threatl):
                    opponentt  = opponentt +1

                    return -10000


                if self.k_in_row(board, pos, player, dir, threatl):
                    playersc += 5

        centerspots = [(3, 3), (2, 3), (3, 2)]
        index = 0

        while index < len(centerspots):
            pos = centerspots[index]
            if pos in board and board[pos] == player:
                centerb += 5
            elif pos in board and board[pos] == opponent:
                centerb -= 3
            index = index +1

        return playersc - 20 * opponentt + centerb

#@staticmethod
    def k_in_row(self, board, pos, player, dir, k):
        """helpe function: Return true if there is a line of k cells in direction dir including position pos on board for player."""
        (delta_x, delta_y) = dir
        x, y = pos
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = pos
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= k



