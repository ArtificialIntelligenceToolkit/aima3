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

    def __init__(self, game, policy_value_fn, c_puct=5, n_playout=10000):
        """
        Arguments:
        policy_value_fn -- a function that takes in a game and board state and outputs a list of (action, probability)
            tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from
            the current player's perspective) for the current player.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more
        """
        self.game = game
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
            state = self.game.result(state, action)
        # Evaluate the leaf using a network which outputs a list of (action, probability)
        # tuples p and also a score v in [-1, 1] for the current player.

        # Check for end of game.
        end = self.game.terminal_test(state)
        if not end:
            action_probs, leaf_value = self.policy(self.game, state)
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            leaf_value = self.game.utility(state, self.game.to_move(state))
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
        allowed_actions = self.game.actions(state)
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy)
        # calc the move probabilities based on the visit counts at the root node
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()
                      if act in allowed_actions]
        if len(act_visits) == 0: ## HOW? Childred, but no valid actions?
            if len(allowed_actions) == 0:
                acts, act_probs = [], [] ## No possible moves!
            else:
                acts = allowed_actions
                act_probs = [1/len(allowed_actions) for i in range(len(allowed_actions))]
        else:
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
