from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from robot_env import RobotEnv  
from gymnasium.wrappers import TimeLimit
import os

from gymnasium.wrappers import TimeLimit # 追加

def make_env(rank):
    def _init():
        env = RobotEnv(xml_name="pendulum.xml")
        # 1000ステップ（10秒分など）で強制的に区切る設定を追加
        env = TimeLimit(env, max_episode_steps=1000) 
        return env
    return _init

if __name__ == "__main__":
    # 8つのプロセスを立ち上げて並列化（CPUのコア数に合わせて4〜12くらいが目安）
    env = SubprocVecEnv([make_env(i) for i in range(8)]) 

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=256, 
        device="cuda",
        tensorboard_log="./ppo_robot_logs/"
    )

    print("GPUで学習を開始します。")
    model.learn(total_timesteps=800000) 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "..", "ppo_inverted_pendulum")
    model.save(save_path)
    print("学習が完了しました！")