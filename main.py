from matplotlib import pyplot as plt
from environment import BlackJackEnv


def main():
    choice = int(input("Enter: \n(1) for preset parameters\n(2) to set a new environment\n"))
    if choice == 1:
        preset()
    else:
        params()


def preset():
    """
    To run experiments and show graphs as per report parameters
    """
    # parameters
    alpha = 0.01
    gamma = 1  # discount factor stays 1 since games are with replacement
    epsilon = 0.2
    c = 1
    num_episodes = 1000
    num_experiments = 100

    selection_types = [1, 2, 3]
    total_wins_q = []
    total_losses_q = []
    total_rewards_q = []
    total_wins_sarsa = []
    total_losses_sarsa = []
    total_rewards_sarsa = []
    for selection_type in selection_types:
        env = BlackJackEnv(alpha, gamma, epsilon, c, 1, selection_type)
        wins, losses, rewards = env.run_simulation(num_episodes, num_experiments)
        total_wins_q.append(wins)
        total_losses_q.append(losses)
        total_rewards_q.append(rewards)
    plot_preset(total_wins_q, total_losses_q, total_rewards_q, num_episodes, num_experiments, 1)

    for selection_type in selection_types:
        env = BlackJackEnv(alpha, gamma, epsilon, c, 2, selection_type)
        wins, losses, rewards = env.run_simulation(num_episodes, num_experiments)
        total_wins_sarsa.append(wins)
        total_losses_sarsa.append(losses)
        total_rewards_sarsa.append(rewards)
    plot_preset(total_wins_sarsa, total_losses_sarsa, total_rewards_sarsa, num_episodes, num_experiments, 2)


def plot_preset(total_wins, total_losses, total_rewards, episodes, T, agent_type):
    """
    To show graphs as above defined parameters
    """
    agents = ["e-greedy", "opt init values", "UCB"]
    type = ["Q-Learning", "SARSA"]
    for j in range(3):
        plt.figure(1)
        plt.plot([i / T for i in total_rewards[j]], label=f'{agents[j]}')
        plt.title(f'Average rewards for {T} simulations over {episodes} episodes\nfor {type[agent_type - 1]}')
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Average rewards')

        plt.figure(2)
        plt.title(f'Average wins and losses for {T} simulations over {episodes} episodes\nfor {type[agent_type - 1]}')
        plt.plot([i / T for i in total_wins[j]], label=f"Avg Wins for {agents[j]}")
        plt.plot([i / T for i in total_losses[j]], label=f"Avg Losses for {agents[j]}")
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Percentage')

    plt.show()


def params():
    """
    To run experiments and show graphs as per user input
    """
    agent_type = int(input("Enter: \n(1) for Q-learning agent\n(2) for SARSA agent\n"))
    selection_type = int(
        input("Enter: \n(1) for epsilon greedy action selection\n(2) for optimistic initial values\n(3) for UCB\n"))

    alpha = float(input("Enter learning rate (alpha): "))

    if selection_type == 1:
        epsilon = float(input("Enter epsilon value: "))
    else:
        epsilon = 0

    if selection_type == 3:
        c = float(input("Enter the desired amount of exploration (c): "))
    else:
        c = 0

    # discount factor stays 1 since games are with replacement
    gamma = 1.0

    # play games num_episodes times
    num_episodes = int(input("Enter number of games to play: "))
    # run num_experiments experiments (each experiment has num_episode rounds)
    num_experiments = int(input("Enter number of simulations: "))

    env = BlackJackEnv(alpha, gamma, epsilon, c, agent_type, selection_type)
    wins, losses, rewards = env.run_simulation(num_episodes, num_experiments)
    env.plot(num_episodes, num_experiments, rewards, wins, losses)


if __name__ == "__main__":
    main()
