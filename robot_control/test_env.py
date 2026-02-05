import numpy as np
from robot_env import RobotEnv  # あなたの環境ファイル名に合わせてください

def test_robot_environment():
    print("=== Robot Environment Physical Test ===")
    
    # 1. 環境の初期化
    try:
        env = RobotEnv(xml_name="pendulum.xml")
        print("Successfully loaded the environment.")
    except Exception as e:
        print(f"Error loading environment: {e}")
        return

    # 2. リセット
    obs, _ = env.reset()
    print(f"\nInitial Observation (Reset):")
    print(f"  Roll Angle: {obs[0]:.4f} rad")
    print(f"  L-Wheel Vel: {obs[3]:.4f}")
    print(f"  R-Wheel Vel: {obs[4]:.4f}")

    # 3. 数ステップ実行して挙動を確認
    print("\n=== Running 20 Steps with Random Actions ===")
    for i in range(20):
        # ランダムなアクション [-1.0, 1.0]
        action = env.action_space.sample()
        
        # 1ステップ実行
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 内容の表示
        print(f"Step {i:02d}:")
        print(f"  Action (L/R): [{action[0]:.3f}, {action[1]:.3f}]")
        print(f"  Obs -> Roll: {obs[0]:.4f} rad, RollVel: {obs[1]:.4f} rad/s")
        print(f"  Obs -> WheelVel (L/R): {obs[3]:.4f} / {obs[4]:.4f}")
        print(f"  Reward: {reward:.4f}")
        
        # 相補フィルターの発散チェック
        if np.isnan(obs[0]) or np.isinf(obs[0]):
            print("\n[!!!] FATAL ERROR: Filtered Roll is NaN or Inf!")
            print("Check if self.alpha is between 0 and 1 (e.g., 0.98).")
            break
            
        if terminated:
            print("  --- Robot Tipped Over ---")
            env.reset()

    print("\n=== Test Finished ===")

if __name__ == "__main__":
    test_robot_environment()