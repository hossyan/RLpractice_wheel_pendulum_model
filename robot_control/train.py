from stable_baselines3 import PPO
from robot_env import RobotEnv  
import os

# 1. 環境の生成
env = RobotEnv(xml_name="pendulum.xml")

# 2. PPOモデルの設定
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.0003, # 学習率
    device="auto"         # GPUがあればGPU、なければCPUを自動選択
)

# 3. 学習開始
print("学習を開始します。")
model.learn(total_timesteps=1000000) # 100_000回試す(10ms * 100,000 = 1000s)

# 4. 学習済みモデル（重み）の保存
model.save("ppo_inverted_pendulum")
print("学習が完了し、モデルを保存しました！")