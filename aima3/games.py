"""Games, or Adversarial Search (Chapter 5)"""

from collections import namedtuple
import itertools
import random

import numpy as np

from .utils import argmax
from .mcts import MCTS

infinity = float('inf')
GameState = namedtuple('GameState', 'to_move, utility, board, moves')

# ______________________________________________________________________________
# Minimax Search

def minimax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minimax_decision:
    return argmax(game.actions(state),
                  key=lambda a: min_value(game.result(state, a)))

# ______________________________________________________________________________


def alphabeta_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = game.to_move(state)

    # Functions used by alphabeta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_cutoff_search:
    best_score = -infinity
    beta = infinity
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def alphabeta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    # Functions used by alphabeta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a),
                                 alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a),
                                 alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or
                   (lambda state, depth: depth > d or
                    game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -infinity
    beta = infinity
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action

# ______________________________________________________________________________
# Some Sample Games


class Game():
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def reset(self):
        pass

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

    def play_tournament(self, matches, *players, mode="random", verbose=0, **kwargs):
        """
        mode -
          "random" - randomly select who gos first
          "ordered" - play in order given
          ""one-each" - play each pairing twice, changing who goes first
        """
        results = {"DRAW": 0}
        for player in players:
            results[player.name] = 0
        pairings = [(a[1],b[1]) for (a,b) in
                    itertools.product(enumerate(players), enumerate(players))
                    if a[0] != b[0]]
        if verbose: print("Tournament to begin with %s matches..." % len(pairings))
        for (p1, p2) in pairings:
            if mode == "one-each":
                result = self.play_matches(matches, p1, p2, flip_coin=False, verbose=verbose, **kwargs)
                for player_name in result:
                    results[player_name] += result[player_name]
                result = self.play_matches(matches, p2, p1, flip_coin=False, verbose=verbose, **kwargs)
                for player_name in result:
                    results[player_name] += result[player_name]
            else:
                result = self.play_matches(matches, p1, p2, flip_coin=(mode=="random"), verbose=verbose, **kwargs)
                for player_name in result:
                    results[player_name] += result[player_name]
        return results

    def play_matches(self, matches, *players, flip_coin=True, verbose=0, **kwargs):
        results = {"DRAW": 0}
        for player in players:
            results[player.name] = 0
        for i in range(matches):
            result = self.play_game(*players, flip_coin=flip_coin, verbose=verbose, **kwargs)
            for player_name in result:
                results[player_name] += 1
        return results

    def get_action(self, player, state, turn, **kwargs):
        """
        Level of indirection for overriding this method.
        """
        move = player.get_action(state, turn, **kwargs)
        return move

    def play_game(self, *players, flip_coin=True, verbose=1, **kwargs):
        """Play an n-person, move-alternating game."""
        if len(players) == 0:
            raise Exception("Need at least 1 player")
        self.reset()
        state = self.initial
        if flip_coin:
            players = list(players)
            random.shuffle(players)
        for player in players:
            move = player.set_game(self) ## do initialization, reset here
        turn = 1
        while True:
            for player in players:
                if verbose:
                    print("%s is thinking..." % player.name)
                move = self.get_action(player, state, turn, **kwargs)
                state = self.result(state, move)
                if verbose:
                    print("%s makes action %s:" % (player.name, move))
                    self.display(state)
                if self.terminal_test(state):
                    result = self.utility(state, self.to_move(self.initial))
                    self.final_utility = result
                    self.final_state = state
                    if result == 1:
                        retval = [players[0].name]
                    elif result == -1:
                        retval = [p.name for p in players[1:]]
                    elif result == 0:
                        retval = ["DRAW"]
                    if verbose:
                        print("***** %s wins!" % ",".join(retval))
                    return retval
            turn += 1

class Fig52Game(Game):
    """The game represented in [Figure 5.2]. Serves as a simple test case."""

    succs = dict(A=dict(a1='B', a2='C', a3='D'),
                 B=dict(b1='B1', b2='B2', b3='B3'),
                 C=dict(c1='C1', c2='C2', c3='C3'),
                 D=dict(d1='D1', d2='D2', d3='D3'))
    utils = dict(B1=3, B2=12, B3=8, C1=2, C2=4, C3=6, D1=14, D2=5, D3=2)
    initial = 'A'

    def actions(self, state):
        return list(self.succs.get(state, {}).keys())

    def result(self, state, move):
        return self.succs[state][move]

    def utility(self, state, player):
        if player == 'MAX':
            return self.utils[state]
        else:
            return -self.utils[state]

    def terminal_test(self, state):
        return state not in ('A', 'B', 'C', 'D')

    def to_move(self, state):
        return 'MIN' if state in 'BCD' else 'MAX'


class Fig52Extended(Game):
    """Similar to Fig52Game but bigger. Useful for visualisation"""

    succs = {i:dict(l=i*3+1, m=i*3+2, r=i*3+3) for i in range(13)}
    utils = {}
    initial = 1

    def actions(self, state):
        return sorted(list(self.succs.get(state, {}).keys()))

    def result(self, state, move):
        return self.succs[state][move]

    def utility(self, state, player):
        if player == 'MAX':
            return self.utils.get(state, 0)
        else:
            return -self.utils.get(state, 0)

    def terminal_test(self, state):
        return state not in range(13)

    def to_move(self, state):
        return 'MIN' if state in {1, 2, 3} else 'MAX'

class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""

    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, self.h + 1)
                 for y in range(1, self.v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return +1 if player == 'X' else -1
        else:
            return 0

    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player."""
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k

class ConnectFour(TicTacToe):
    """A TicTacToe-like game in which you can only make a move on the bottom
    row, or in a square directly above an occupied square.  Traditionally
    played on a 7x6 board and requiring 4 in a row."""

    def __init__(self, h=7, v=6, k=4):
        TicTacToe.__init__(self, h, v, k)

    def actions(self, state):
        return [(x, y) for (x, y) in state.moves
                if y == 1 or (x, y - 1) in state.board]

    def display(self, state):
        board = state.board
        for y in range(self.v + 1, 0, -1):
            for x in range(1, self.h + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

# ______________________________________________________________________________
# Players for Games

class Player():
    """
    """
    def __init__(self, name):
        self.name = name

    def set_game(self, game):
        ## do init or reset business here in overload
        self.game = game

    def get_action(self, state, turn):
        raise NotImplementedError()

class QueryPlayer(Player):
    """
    """
    def get_action(self, state, turn):
        """Make a move by querying standard input."""
        print("current state:")
        self.game.display(state)
        print("available moves: {}".format(self.game.actions(state)))
        print("")
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
        return move

class RandomPlayer(Player):
    def get_action(self, state, turn):
        """A player that chooses a legal move at random."""
        return random.choice(self.game.actions(state))

class AlphaBetaPlayer(Player):
    def get_action(self, state, turn):
        return alphabeta_search(state, self.game)

class MiniMaxPlayer(Player):
    def get_action(self, state, turn):
        return minimax_decision(state, self.game)

class AlphaBetaCutoffPlayer(Player):
    def get_action(self, state, turn):
        return alphabeta_cutoff_search(state, self.game, d=4,
                                       cutoff_test=None, eval_fn=None)

class MCTSPlayer(Player):
    """AI player based on MCTS"""
    def __init__(self, name, n_playout=20, random_turns=2,
                 c_puct=5, is_selfplay=True):
        super().__init__(name)
        self.n_playout = n_playout
        self.random_turns = random_turns
        self.c_puct = c_puct
        self.is_selfplay = is_selfplay

    def policy(self, game, state):
        """
        A function that takes in a board state and outputs a list of
        (action, probability) tuples and also a score in [-1, 1]
        (i.e. the expected value of the end game score from the
        current player's perspective) for the current player.
        """
        value = game.utility(state, game.to_move(state))
        actions = game.actions(state)
        if len(actions) == 0:
            return [], value
        else:
            prob = 1/len(actions)
            return [(action, prob) for action in actions], value

    def set_game(self, game):
        self.game = game
        self.mcts = MCTS(self.game, self.policy, self.c_puct, self.n_playout)

    def get_action(self, state, turn, return_prob=0):
        ## can make temp based on turn #
        if turn <= self.random_turns:
            temp = 0.1
        else:
            temp = 1e-3
        sensible_moves = self.game.actions(state)
        all_moves = self.game.actions(self.game.initial)
        move_probs = {key: 0.0 for key in all_moves} # the pi vector returned by MCTS as in the alphaGo Zero paper
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(state, temp)
            #print("acts", acts, "probs", probs)
            move_probs.update({key: val for (key,val) in zip(acts, probs)})
            if self.is_selfplay:
                #move_index = np.argmax(probs)
                #move_index = np.random.choice(range(len(acts)), p=probs)
                # add Dirichlet Noise for exploration (needed for self-play training)
                move_index = np.random.choice(range(len(acts)), p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                move = acts[move_index]
                self.mcts.update_with_move(move) # update the root node and reuse the search tree
            else:
                # with the default temp=1e-3, this is almost equivalent to choosing the move with the highest prob
                move_index = np.random.choice(range(len(acts)), p=probs)
                move = acts[move_index]
                # reset the root node
                self.mcts.update_with_move(-1)
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

players = [
    RandomPlayer("Random"),
    AlphaBetaPlayer("AlphaBeta"),
    MiniMaxPlayer("Minimax"),
    AlphaBetaCutoffPlayer("AlphaBetaCutoff"),
    MCTSPlayer("MonteCarloTreeSearch")
]

# game = TicTacToe()
# game.play_tournament(1, *players, verbose=1)
