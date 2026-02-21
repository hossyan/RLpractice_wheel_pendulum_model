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

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


if __name__ == "__main__":
    env = SubprocVecEnv([make_env(i) for i in range(8)]) 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.join(script_dir, "..", "ppo_inverted_pendulumV2.zip")
    model = PPO.load(load_path, env=env, device="cuda")

    policy_kwargs = dict(
        log_std_init=-0.5,  # ここで初期のばらつきを指定
    )

    # model = PPO(
    #     "MlpPolicy", 
    #     env, 
    #     verbose=1,
    #     # learning_rate=linear_schedule(3e-4),
    #     learning_rate=3e-4,
    #     ent_coef=0.01,
    #     n_steps=256, 
    #     clip_range=0.3,
    #     max_grad_norm=0.3,
    #     policy_kwargs=policy_kwargs,
    #     device="cpu",
    #     tensorboard_log="./logs2/"
    # )

    print("学習を開始します。")
    model.learn(total_timesteps=800000) 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "..", "ppo_inverted_pendulumV3")
    model.save(save_path)
    print("学習が完了しました！")