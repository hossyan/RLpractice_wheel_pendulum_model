from stable_baselines3 import PPO
from robot_env import RobotEnv
import mujoco.viewer
import time
import numpy as np 
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "ppo_inverted_pendulum.zip")

env = RobotEnv(xml_name="pendulum.xml")
model = PPO.load(model_path, device="cuda")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    obs, info = env.reset()    
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        print(f"Action (L/R): {action}")
        
        if np.random.rand() < 0.02:
            push_force = np.random.uniform(-2.0, 2.0)
            env.unwrapped.data.xfrc_applied[env.unwrapped.body_id, 1] = push_force
        else:
            env.unwrapped.data.xfrc_applied[env.unwrapped.body_id, 1] = 0.0

        obs, reward, terminated, truncated, info = env.step(action)
        
        viewer.sync()
        
        time.sleep(0.01)
        
        # 転んだらリセット
        if terminated:
            obs, info = env.reset()
            print("転倒したためリセットしました")