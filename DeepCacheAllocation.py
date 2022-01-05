import numpy as np

import gym
from gym import spaces

MAX_CACHE_CAPACITY = 30  # cache capacity - limited to 30
INITIAL_X1_SLOT = 15  #


class DeepCacheNetw(gym.Env):
    """Deep QN Environment for Cache Allocation"""
    metadata = {'render.modes': ['human']}

    def __init__(self, x1):
        super(DeepCacheNetw, self).__init__()
        self.x1 = x1
        self.x2 = 30 - x1
        self.reward_range = (-20, 0)  # -20 the worst case of all requests
        self.action_space = spaces.Discrete(3)

        # The range in which x1 can take value
        self.observation_space = spaces.Box(
            low=-20, high=0, shape=(1, 30), dtype=np.float16)

    # first count the cost, and then calculate the reward (that is the minus cost)
    def step(self, action):

        penalty = 0

        # if the value of x1 + *action it takes* is negative or over the max_cache_capacity it gains a penalty
        if self.x1 + action < 0 or self.x2 - action < 0 or self.x1 + action > MAX_CACHE_CAPACITY:
            penalty = 300
        else:
            self.x1 = self.x1 + action
            self.x2 = self.x2 - action

        # Now, let's calculate the reward

        rand_reqSP1 = np.random.randint(low=1, high=100, size=1)  # 10 random requests of sp1 (numbers from 1 to 100)
        rand_reqSP2 = np.random.randint(low=1, high=200, size=100)  # 10 random requests of sp2 (numbers from 1 to 200)

        # everytime the requests is not in cache we increase the size of the instantaneous cost

        inst_cost = 0

        for req in rand_reqSP1:
            if req > self.x1:
                inst_cost += 1

        for req in rand_reqSP2:
            if req > self.x2:
                inst_cost += 1

        reward = -inst_cost - penalty

        observation = self.x1

        print('observation, reward = ', observation, reward)  # stamps every step x1 and the reward


        return observation, reward, True, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.x1 = INITIAL_X1_SLOT

        observation = self.x1

        return observation