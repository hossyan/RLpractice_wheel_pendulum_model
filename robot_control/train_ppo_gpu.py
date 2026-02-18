from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from robot_env import RobotEnv  
from gymnasium.wrappers import TimeLimit
import os
from stable_baselines3.common.monitor import Monitor

def make_env(rank):
    def _init():
        env = RobotEnv(xml_name="pendulum.xml")
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps=2000) 
        return env
    return _init


if __name__ == "__main__":
    env = SubprocVecEnv([make_env(i) for i in range(8)]) 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # load_path = os.path.join(script_dir, "..", "ppo_inverted_pendulum.zip")
    # model = PPO.load(load_path, env=env, device="cuda")

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=256, 
        device="cuda",
        tensorboard_log="./logs/"
    )

    print("GPUで学習を開始します。")
    model.learn(total_timesteps=800000) 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "..", "ppo_inverted_pendulum")
    model.save(save_path)
    print("学習が完了しました！")