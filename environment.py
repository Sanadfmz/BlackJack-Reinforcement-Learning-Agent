import gym
import numpy as np
from matplotlib import pyplot as plt

from agent import Agent

"""
    Defines a black jack environment with an agent playing the game
    Parameters:
                -alpha: learning rate of agent
                -gamma: discount factor of agent
                -epsilon: probability of selecting random action instead of the 'optimal' action
                -agent_type: 1 for q_learning agent and 2 for SARSA agent
                -selection_type: 1 for epsilon-greedy, 2 for OIV and 3 for UCB
"""


class BlackJackEnv:
    def __init__(self, alpha, gamma, epsilon, c, agent_type, selection_type):
        self.env = gym.make('Blackjack-v1')
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.c = c
        self.agent_type = agent_type
        self.selection_type = selection_type

    def run_simulation(self, episodes, T):
        """"
        Run simulations to observe learning
        """
        wins = [0] * episodes
        losses = [0] * episodes
        rewards = [0] * episodes

        for t in range(T):
            # reset each experiment
            win = 0
            loss = 0
            reward_total = 0
            agent = Agent(self.env, self.alpha, self.gamma, self.epsilon, self.c, self.env.action_space,
                          self.agent_type, self.selection_type)

            # for each episode...
            for episode in range(episodes):
                done = False
                # initialise s
                current_observation = self.env.reset()

                # for SARSA initialise current action
                if self.agent_type == 2:
                    current_action = agent.choose_action(current_observation, episode)

                # for each step in episode...do while game is not over
                while not done:
                    # SARSA as per Algorithm 2
                    if self.agent_type == 2:
                        next_observation, reward, done, _ = self.env.step(current_action)
                        next_action = agent.choose_action(next_observation, episode)
                        agent.updateSarsa(current_observation, current_action, reward, next_observation, next_action)
                        current_action = next_action
                        current_observation = next_observation
                        
                    # Q-Learning as per Algorithm 1
                    else:
                        action = agent.choose_action(current_observation, episode)
                        next_observation, reward, done, _ = self.env.step(action)
                        agent.updateQL(current_observation, action, reward, next_observation)
                        current_observation = next_observation

                    reward_total += reward


                # after current round is done
                if reward > 0:
                    win += 1
                else:
                    loss += 1

                wins[episode] += (1.0 * win) / (episode + 1)
                losses[episode] += (1.0 * loss) / (episode + 1)
                rewards[episode] += (1.0 * reward_total) / (episode + 1)
                
        self.env.close()
        return wins, losses, rewards

    def plot(self, episodes, T, rewards, wins, losses):
        """
        plot graphs from results
        """
        plt.figure(1)
        plt.plot([i / T for i in rewards], label='Avg rewards')
        plt.title(f'Average rewards for {T} simulations over {episodes} episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Average rewards')

        plt.figure(2)
        plt.title(f'Average wins and losses for {T} simulations over {episodes} episodes')
        plt.plot([i / T for i in wins], label="Avg Wins")
        plt.plot([i / T for i in losses], label="Avg Losses")
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Percentage')

        plt.show()
