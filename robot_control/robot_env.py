import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

class robot_env(gym.Env):
    def __init__(self, xml_name="pendulum.xml"):
        super().__init__()
        
        # --- 1. MuJoCoモデルの読み込み ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(script_dir, xml_name)

        # モデルとデータの読み込み
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.l_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_L")
        self.r_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_R")  

        # --- 2. アクション空間 (Action Space) の定義 ---
        # [右のトルク, 左のトルク] 
        low_act = np.array([-0.021, -0.021], dtype=np.float32)
        high_act = np.array([0.021, 0.021], dtype=np.float32)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

        # --- 3. 観測空間 (Observation Space) の定義 ---
        # [本体ピッチ角, 本体角速度, 右車輪角速度, 左車輪角速度] の4つとします
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(4,), 
            dtype=np.float32
        )

        # --- 4. 制御周期の設定 ---
        self.dt = self.model.opt.timestep

    def _get_obs(self):
        # --- 1. 車体の角度（ロール角）の計算 ---
        quat = self.data.xquat[self.body_id]
        w, x, y, z = quat
        roll_rad = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))

        # --- 2. その他のデータの取得 ---
        body_ang_vel = self.data.qvel[3] 
        l_wheel_vel = self.data.qvel[self.l_wheel_id]
        r_wheel_vel = self.data.qvel[self.r_wheel_id]

        return np.array([roll_rad, body_ang_vel, l_wheel_vel, r_wheel_vel], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

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