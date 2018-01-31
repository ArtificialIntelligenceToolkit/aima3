"""

Based on various code examples, including:

https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py

"""

import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class Node(object):
    """
    A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """
    def __init__(self, parent, P):
        self.parent = parent
        self.children = {}  # a map from action to Node
        self.n_visits = 0
        self.P = P
        self.Q = 0
        self.u = 0

    def expand(self, action_priors):
        """
        Expand tree by creating new children.

        action_priors -- output from policy function - a list of tuples of actions
            and their prior probability according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self, c_puct):
        """
        Select action among children that gives maximum action value, Q plus bonus u(P).

        Returns:
            A tuple of (action, next_node)
        """
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        Update node values from leaf evaluation.

        Arguments:
            leaf_value -- the value of subtree evaluation from the current player's perspective.
        """
        # Count visit.
        self.n_visits += 1
        # Update Q, a running average of values for all visits.
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        """
        Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u

        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        self.u = c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u

    def is_leaf(self):
        """
        Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self.children == {}

    def is_root(self):
        return self.parent is None

class MCTS(object):
    """
    A simple implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        Arguments:
        policy_value_fn -- a function that takes in a board state and outputs a list of (action, probability)
            tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from
            the current player's perspective) for the current player.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more
        """
        self.root = Node(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        node = self.root
        while (1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self.c_puct)
            state.do_move(action)
        # Evaluate the leaf using a network which outputs a list of (action, probability)
        # tuples p and also a score v in [-1, 1] for the current player.
        action_probs, leaf_value = self.policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == 2:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        Runs all playouts sequentially and returns the available actions and their corresponding probabilities
        Arguments:
            state -- the current state, including both game state and the current player.
            temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
            the available actions and the corresponding probabilities
        """
        allowed_actions = state.get_allowed_actions()
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy)
        # calc the move probabilities based on the visit counts at the root node
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()
                      if act in allowed_actions]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        Step forward in the tree, keeping everything we already know about the subtree.
        """
        #print("Updating with last_move:", last_move, self.root.children)
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, name, policy_value_function, random_turns=1, c_puct=5, n_playout=1000,
                 is_selfplay=True):
        self.name = name
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self.is_selfplay = is_selfplay
        self.random_turns = random_turns

    def reset(self):
        self.mcts.update_with_move(-1)

    def get_action(self, state, turn, return_prob=0):
        ## can make temp based on turn #
        if turn <= self.random_turns:
            temp = 0.1
        else:
            temp = 1e-3
        sensible_moves = state.get_allowed_actions()
        move_probs = state.get_zero_probs() # the pi vector returned by MCTS as in the alphaGo Zero paper
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(state, temp)
            #print("acts", acts, "probs", probs)
            move_probs[list(acts)] = probs
            if self.is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                #move = acts[np.argmax(probs)]
                move = np.random.choice(acts, p=probs)
                #move = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                self.mcts.update_with_move(move) # update the root node and reuse the search tree
            else:
                # with the default temp=1e-3, this is almost equivalent to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
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
