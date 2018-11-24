# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util
import sys

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    self.best_actions = {} # Stores best actions in order to gain performance
    for state in self.mdp.getStates():
        self.best_actions[state] = None

    # We compute the value of each state
    # We also store the best action to take for each step
    for i in range(self.iterations):
        next_values = self.values.copy()
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                best_action = None
                best_score = -sys.float_info.max
                for a in self.mdp.getPossibleActions(s):
                    e = self.getQValue(s, a)
                    if e > best_score:
                        best_score = e
                        best_action = a
                self.best_actions[s] = best_action
                next_values[s] = best_score
        self.values = next_values

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    sum_ = 0
    # We apply the formula of the notes
    for s_prime, t in self.mdp.getTransitionStatesAndProbs(state, action):
        r = self.mdp.getReward(state, action, s_prime)
        sum_ += t * (r + self.discount*self.values[s_prime])
    return sum_

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    return self.best_actions[state]

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
