import numpy as np

T = 10000
N = 10
true_rewards = np.random.uniform(low=0, high=1, size=N)
estimated_rewards = np.zeros(N)
chosen_cnt = np.zeros(N)
total_reward = 0

def calculate_delta(T, item):
    if chosen_cnt[item] == 0: # 当没有选过的时候就乐观的认为一定会喜欢
        return 1
    else:
        return np.sqrt(2 * np.log(T) / chosen_cnt[item])

def UCB(t, N):
    upper_bound_probs = [estimated_rewards[item] + calculate_delta(t,item) for item in range(N)] # 充分利用历史信息进行选择
    item = np.argmax(upper_bound_probs) # 随着时间的推移, 有一些选的次数少的 p 会变大
    reward = np.random.binomial(n=1, p=true_rewards[item])
    return item, reward

for t in range(1, T):
    item, reward = UCB(t, N)
    total_reward += reward
    estimated_rewards[item] = ((t - 1) * estimated_rewards[item] + reward) / t
    chosen_cnt[item] += 1

print(estimated_rewards)
print(total_reward)