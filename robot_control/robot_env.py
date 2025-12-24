import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class robot_env(gym.Env):
    def __init__(self, model_path):
        super(robot_env, self).__init__()
        
        # 1. MuJoCoモデルの読み込み準備
        # ここでXMLファイルを読み込む設定をします
        
        # 2. アクション空間 (Action Space) の定義
        # ロボットが「何ができるか」を定義します
        # タイヤ2つへのトルク指令（例：-1.0 から 1.0 の範囲）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 3. 観測空間 (Observation Space) の定義
        # ロボットが「何を見ることができるか」を定義します
        # 例：傾き、角速度、タイヤの回転角、回転速度など
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # 練習を最初からやり直す（ロボットを初期位置に戻す）処理
        super().reset(seed=seed)
        # 状態を初期化し、最初の「観測値」を返す必要があります
        observation = np.zeros(6, dtype=np.float32)
        return observation, {}

    def step(self, action):
        # 1コマ（例えば0.01秒）時間を進める処理
        # a. action（モーター出力）をMuJoCoに伝える
        # b. 物理シミュレーションを1ステップ進める
        # c. 新しい状態（観測値）を取得する
        # d. 報酬（reward）を計算する
        # e. 転んだかどうかの判定（terminated）をする
        
        observation = np.zeros(6, dtype=np.float32) # 仮の観測値
        reward = 0.0                                # 仮の報酬
        terminated = False                          # 終了判定
        truncated = False                           # 時間切れ判定
        
        return observation, reward, terminated, truncated, {}