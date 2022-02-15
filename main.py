from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from DeepCacheAllocation import DeepCacheNetw
# from env.DeepCacheAllocation import DeepCacheAllocation

import ctypes

hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cudart64_100.dll")

max_cost_SP1 = 200
max_cost_SP2 = 300

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: DeepCacheNetw(90, max_cost_SP1, max_cost_SP2)])

model = PPO('MlpPolicy', env, verbose=1)
print("STARTING TO LEARN")
model.learn(total_timesteps=50000)
#try again with 1.2 and 1.2 and the same timesteps
print("STARTING TO PREDICT")
obs = env.reset()

observation_list = []
reward_list = []
for time_step in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    # making SP1 plot
    observation_list.append(obs[0][0][0])
    # making reward plot
    reward_list.append(rewards[0])

    with open("x1.txt", "a") as o:
        o.write("\n")
        o.write("\n".join(str(x) for x in obs))
        o.write("\n")

    with open("reward.txt", "a") as o:
        o.write("\n")
        o.write("\n".join(str(x) for x in rewards))
        o.write("\n")

plt.scatter(observation_list, reward_list)

# plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
plt.title('SP1 and its reward')
plt.xlabel('SP1- value')
plt.ylabel('REWARD - value')
plt.show()
