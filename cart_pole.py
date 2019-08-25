import numpy as np
import gym
import time

def get_weights_by_hill_climbing(best_weights):
# 通过爬山算法选取权值(在当前最好权值上加上随机值)
    return best_weights + np.random.normal(0.0, 0.1, 5)

def get_weights_by_random_guess():
# 选取随机猜测的5个随机权值
    return np.random.rand(5)

def get_action(weights, observation):
# 根据权值对当前状态做出决策
    if weights[:4].dot(observation) + weights[4] >= 0:
        return 1
    else:
        return 0

def get_sum_reward_by_cur_weights(env, weights):
# 测试不同权值的控制模型有效控制的持续时间(也叫奖励)
    observation = env.reset()  # 重置初始状态
    sum_reward = 0  # 记录总的奖励
    for t in range(1000):
        time.sleep(0.01)
        env.render()
        action = get_action(weights, observation)
        # 执行动作并获取这一动作下的下一时间步长状态
        observation, reward, done, info = env.step(action)
        sum_reward += reward
        # 若游戏结束，则返回
        if done:
            break
    return sum_reward

def get_best_result(algorithm='random_guess'):
# 默认是随机算法
    env = gym.make('CartPole-v1')
    best_reward = 0  # 初始化最佳奖励
    best_weights = np.random.rand(5)

    for iter in range(10000):  # 迭代次数为一万次，表示给你一万次机会去让奖励大于200
        cur_weights = None

        if algorithm == 'hill-climbing':  # 选取动作决策的算法
            cur_weights = get_weights_by_hill_climbing(best_weights)
        else:  # 若为随机猜测算法，则选取随机权值
            cur_weights = get_weights_by_random_guess()
        
        # 获取当前权值的模型控制的奖励和
        cur_sum_reward = get_sum_reward_by_cur_weights(env, cur_weights)
        # 更新当前最优权值
        if cur_sum_reward > best_reward:
            best_reward = cur_sum_reward
            best_weights = cur_weights
        # 达到最佳奖励阈值后结束
        if best_reward >= 200:
            break
    print(iter)
    return best_reward, best_weights

# 程序从此处开始执行
print(get_best_result('hill_climbing'))