from .mcts import MCTSPlayer
import numpy as np
import random
import copy

class State():
    def __init__(self, board, playerTurn):
        self.board = copy.copy(board)
        self.playerTurn = playerTurn

    def get_zero_probs(self):
        return np.zeros(len(self.board))

    def do_move(self, action):
        self.board[action] = self.playerTurn
        self.playerTurn = -1 * self.playerTurn

    def get_allowed_actions(self):
        return [pos for pos in range(len(self.board)) if self.board[pos] == 0]

    def game_end(self):
        winners = [[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8],
                   [0, 4, 8],
                   [6, 4, 2],
                   [0, 3, 6],
                   [1, 4, 7],
                   [2, 5, 8]]
        if any([all([self.board[pos] == -self.playerTurn for pos in winner]) for winner in winners]):
            status = -self.playerTurn
        elif any([all([self.board[pos] == self.playerTurn for pos in winner]) for winner in winners]):
            status = self.playerTurn
        elif not any([self.board[pos] == 0 for pos in range(len(self.board))]):
            status = 2
        else:
            status = 0
        end = status != 0
        return end, status

    def render(self):
        char = {-1: "X", 1: "O", 0: "."}
        print("-+-", end="")
        for col in range(3):
            print("-", end="-+-")
        print()
        for row in range(3):
            print(" | ", end="")
            for col in range(3):
                print(char[self.board[row * 3 + col]], end=" | ")
            print()
            print("-+-", end="")
            for col in range(3):
                print("-", end="-+-")
            print()
        print()

    def get_current_player(self):
        return self.playerTurn

class TicTacToe():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def tournament(self, rounds=10, verbose=1):
        results = {
            p1.name: 0,
            p2.name: 0,
            "draw": 0,
            }
        for round in range(rounds):
            result = self.play(verbose=verbose)
            results[result] += 1
        return results

    def play(self, who_goes_first=None, verbose=1, verify=True):
        """
        who_goes_first - "p1" or "p2" or None for random
        """
        self.p1.reset()
        self.p2.reset()
        ## assign who goes first:
        ## -1 always goes first
        playerTurn = -1
        if who_goes_first is None:
            who_goes_first = "p1" if random.random() < .5 else "p2"
        if who_goes_first == "p1":
            if verbose: print("%s goes first" % p1.name)
            players = {-1: self.p1, 1: self.p2}
        else:
            if verbose: print("%s goes first" % p2.name)
            players = {1: self.p1, -1: self.p2}
        ## board state at beginning:
        state = State([0] * 9, playerTurn)
        turn = 0
        status = 0
        while status == 0:
            turn = turn + 1
            assert state.playerTurn == playerTurn, "playerTurn is wrong!"
            if verbose: print("%s is thinking..." % players[state.playerTurn].name)
            probs = None
            action, probs = players[state.playerTurn].get_action(state, 0, return_prob=True)
            if verbose: print("%s makes a move: action = %s" % (players[state.playerTurn].name, action))
            if verbose and probs is not None: print("Probabilities:", ",".join([("%.2f" % p) for p in probs]))
            if verify and action not in state.get_allowed_actions():
                raise Exception("Invalid move attempted!")
            state.do_move(action)
            # The next state:
            if verbose: state.render()
            end, status = state.game_end()
            if status == 0:
                playerTurn = -1 * playerTurn
        if status == 2:
            if verbose: print("Draw! The cat wins.")
            return_status = "draw"
        else:
            if verbose: print("%s wins!" % players[status].name)
            return_status = players[status].name
        return return_status

def policy_f(state):
    """
    policy_value_fn -- a function that takes in a board state and outputs a list of (action, probability)
         tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from
         the current player's perspective) for the current player.
    """
    actions = state.get_allowed_actions()
    winners = [[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8],
               [0, 4, 8],
               [6, 4, 2],
               [0, 3, 6],
               [1, 4, 7],
               [2, 5, 8]]
    if any([all([state.board[pos] == -state.playerTurn for pos in winner]) for winner in winners]):
        value = -1
    elif any([all([state.board[pos] == state.playerTurn for pos in winner]) for winner in winners]):
        value = 1
    else:
        value = 0
    if len(actions) == 0:
        return [], value
    else:
        prob = 1/len(actions)
        return [(action, prob) for action in actions], value

class RandomPlayer():
    def __init__(self, name):
        self.name = name

    def reset(self):
        pass

    def get_action(self, state, turn, return_prob):
        actions = state.get_allowed_actions()
        if return_prob:
            return random.choice(actions), [1/len(actions) for i in actions]
        else:
            return random.choice(actions)

#p1 = RandomPlayer("Rando-1")
#p2 = RandomPlayer("Rando-2")
#p1 = MCTSPlayer("Monte-1", policy_f, random_turns=5, n_playout=200)
#p2 = MCTSPlayer("Monte-2", policy_f, random_turns=0, n_playout=10)
#ttt = TicTacToe(p1, p2)
