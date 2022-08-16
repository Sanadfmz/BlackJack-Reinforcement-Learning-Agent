import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

""""
    Defines an agent playing the blackjack game
    Parameters:
                -env: the openai blackjack environment in which the agent plays against one dealer
                -alpha: learning rate of agent
                -gamma: discount factor of agent
                -epsilon: probability of selecting random action instead of the 'optimal' action
                -c: parameter which regulates the rate of exploration for UCB
                -action_space: all possible actions of the agent (0 for stick and 1 for hit)
                -agent_type: 1 for q_learning agent and 2 for SARSA agent
                -selection_type: 1 for epsilon-greedy, 2 for OIV and 3 for UCB  
"""


class Agent:
    def __init__(self, env, alpha, gamma, epsilon, c, action_space, agent_type, selection_type):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.c = c
        self.action_space = action_space
        self.agent_type = agent_type
        self.selection_type = selection_type
        self.valid_actions = list(range(self.env.action_space.n))
        self.opt_values = dict()
        self.Q_table = dict()
        self.action_counter = np.zeros(len(self.valid_actions))

    def create_val_if_new_observation(self, observation):
        """
        Set initial Q values to 0.0 if observation not already in Q table
        Also for OIV set optimistic values
        """
        # for all selection types
        if observation not in self.Q_table:
            self.Q_table[observation] = dict((action, 0.0) for action in self.valid_actions)
        # for OIV
        if self.selection_type == 2:
            if observation not in self.opt_values:
                self.opt_values[observation] = dict((action, 5.0) for action in self.valid_actions)

    def get_maxQ(self, observation):
        """"
        Get max Q value for current observation
        """
        self.create_val_if_new_observation(observation)
        return max(self.Q_table[observation].values())

    def choose_action(self, observation, episode):
        """"
        Choose an action based on selection strategy
        """
        self.create_val_if_new_observation(observation)

        # epsilon greedy
        if self.selection_type == 1: 
            if random.random() > self.epsilon:
                maxQ = self.get_maxQ(observation)
                # tie-breaking if there are more than one actions with max values
                action = random.choice([k for k in self.Q_table[observation].keys()
                                        if self.Q_table[observation][k] == maxQ])
            # take random action
            else:
                action = random.choice(self.valid_actions)

        # OIV
        if self.selection_type == 2:
            maxQ = self.get_maxQ(observation)
            # tie-breaking if there are more than one actions with max values
            action = random.choice([k for k in self.Q_table[observation].keys()
                                    if self.Q_table[observation][k] == maxQ])
            self.action_counter[action] += 1

        # UCB
        if self.selection_type == 3:
            if episode == 0:
                self.action_counter = np.ones(len(self.valid_actions))
            values = list(self.Q_table[observation].values())
            action = np.argmax(values + self.c * np.sqrt((episode + 1) / self.action_counter))
            self.action_counter[action] += 1
        return action

    def updateQL(self, observation, action, reward, next_observation):
        """"
        update Q values as per q learning
        """
        self.Q_table[observation][action] += self.alpha * (reward + (self.gamma * self.get_maxQ(next_observation))
                                                           - self.Q_table[observation][action])
        # for OIV
        if self.selection_type == 2:
            self.update_oiv(observation, action, reward, next_observation)

    def updateSarsa(self, current_state, current_action, reward, next_state, next_action):
        """"
        update Q values as per SARSA
        """
        predicted = self.Q_table[current_state][current_action]
        self.create_val_if_new_observation(next_state)
        target = reward + self.gamma * self.Q_table[next_state][next_action]

        self.Q_table[current_state][current_action] = self.Q_table[current_state][current_action] + self.alpha * (
                target - predicted)

        # for OIV
        if self.selection_type == 2:
            # TODO: update optimistic q-values if using sarsa
            predicted_oiv = self.opt_values[current_state][current_action]
            target_oiv = reward + self.gamma * predicted_oiv
            self.opt_values[current_state][current_action] = predicted_oiv + self.alpha * (target_oiv - predicted_oiv)
            pass

    def update_oiv(self, observation, action, reward, next_observation):
        """"
        update optimistic Q values for OIV
        """

        self.opt_values[observation][action] += self.action_counter[action] * (
                reward + (self.gamma * self.get_maxQ(next_observation))
                - self.opt_values[observation][action])
