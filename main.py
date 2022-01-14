from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from DeepCacheAllocation import DeepCacheNetw
# from env.DeepCacheAllocation import DeepCacheAllocation

import ctypes

hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cudart64_100.dll")

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: DeepCacheNetw(25)])

model = PPO('MlpPolicy', env, verbose=1)
print("STARTING TO LEARN")
model.learn(total_timesteps=5000)

print("STARTING TO PREDICT")
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)

