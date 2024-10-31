import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        """
        Initializes the epsilon-greedy bandit algorithm.
        :param n_arms: Number of items (arms) to recommend.
        :param epsilon: Probability of exploration.
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)  # Number of times each arm has been pulled
        self.values = np.zeros(n_arms)  # Estimated value for each arm

    def select_arm(self):
        """
        Select an arm (recommendation) based on epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            # Explore: randomly choose an arm
            return np.random.randint(0, self.n_arms)
        else:
            # Exploit: choose the arm with the highest estimated value
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        """
        Update the estimated value of the chosen arm based on the reward.
        :param chosen_arm: The arm that was chosen.
        :param reward: The reward received from choosing that arm.
        """
        # Increment the count for the chosen arm
        self.counts[chosen_arm] += 1
        # Calculate the new estimated value of the chosen arm
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update the estimated value using incremental formula for mean
        new_value = value + (reward - value) / n
        self.values[chosen_arm] = new_value

    def recommend(self, n_recommendations):
        """
        Generate a list of recommendations.
        :param n_recommendations: Number of recommendations to make.
        """
        recommendations = []
        for _ in range(n_recommendations):
            arm = self.select_arm()
            recommendations.append(arm)
        return recommendations

# Example Usage
n_arms = 5  # Number of items to recommend
epsilon = 0.1  # Exploration factor
bandit = EpsilonGreedyBandit(n_arms, epsilon)

# Simulate 1000 rounds of recommendations and rewards
for _ in range(1000):
    arm = bandit.select_arm()
    # Simulate a reward (1 for a positive interaction, 0 for no interaction)
    reward = np.random.binomial(1, 0.1 + 0.8 * (arm / (n_arms - 1)))
    bandit.update(arm, reward)

# Generate a list of recommendations
recommendations = bandit.recommend(5)
print("Recommended items:", recommendations)
