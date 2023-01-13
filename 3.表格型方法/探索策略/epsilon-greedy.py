import numpy as np

T = 10000 # 总共进行的轮数
N = 10 # 一共有多少个物品
true_rewards = np.random.uniform(low=0, high=1, size=N)
estimated_rewards = np.zeros(N)
num_of_trials = np.zeros(N)
num_of_rewards = np.zeros(N)
total_reward = 0

def epsilon_greedy(N, epsilon=0.1):
    item = 0
    if np.random.rand() < epsilon: # 随机抽样 exploraton
        item = np.random.randint(low=0, high=N)
    else:
        item = np.argmax(estimated_rewards) # 选最大的 exploitation
    reward = np.random.binomial(n=1, p=true_rewards[item], size=1) # 进行一次抽样(size), 抽样的概率(p), 每次抽样的个数(n)
    return item, reward

for t in range(T):
    item, reward = epsilon_greedy(N)
    total_reward += reward
    num_of_trials[item] += 1
    num_of_rewards[item] += reward
    estimated_rewards[item] = num_of_rewards[item]/num_of_trials[item]

print(estimated_rewards)
print(total_reward)