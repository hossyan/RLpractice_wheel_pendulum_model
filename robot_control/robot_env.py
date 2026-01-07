import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

class RobotEnv(gym.Env):
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
        low_act = np.array([-1.0, -1.0], dtype=np.float32)
        high_act = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

        # --- 3. 観測空間 (Observation Space) の定義 ---
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(5,), 
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
        body_roll_vel = self.data.qvel[3] 
        body_yaw_vel = self.data.qvel[5] 
        l_wheel_vel = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        r_wheel_vel = self.data.qvel[self.model.jnt_dofadr[self.r_wheel_id]]

        return np.array([roll_rad, body_roll_vel, body_yaw_vel, l_wheel_vel, r_wheel_vel], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.data.ctrl[0] = action[0] * 0.021 # wheel_hinge_left
        self.data.ctrl[1] = action[1] * 0.021 # wheel_hinge_right

        # 10ms ごとに学習
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # 終了判定 45度(0.78rad)より傾くと終了
        roll = obs[0] 
        terminated = bool(abs(roll) > 0.78)

        # 報酬
        reward = float(
            -3.0 * obs[0]**2    # 傾きペナルティ
            -1.0 * obs[1]**2    # 揺れペナルティ
            -2.0 * obs[2]**2    # その場回転ペナルティ
            -0.05 * obs[3]**2    # タイヤのスピードペナルティ
            -0.05 * obs[4]**2    # タイヤのスピードペナルティ
            # -1.0 * (abs(action[0]) + abs(action[1]))  # 電流値ペナルティ
            +5.0 * (abs(obs[0]) < 0.0872) # 倒立報酬(5度以内)
        )

        truncated = False     # 時間切れならTrue
        info = {}             # おまけ情報

        return obs, reward, terminated, truncated, info