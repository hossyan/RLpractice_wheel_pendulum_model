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
        low_act = np.array([-1.0], dtype=np.float32)
        high_act = np.array([1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

        # --- 3. 観測空間 (Observation Space) の定義 ---
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(5,), 
            dtype=np.float32
        )

        # --- 4. 制御系の設定 ---
        self.dt = self.model.opt.timestep
        self.filtered_roll = 0.0
        self.alpha = 0.6

        # --- 5. 報酬設定用変数の初期化 ---
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _get_obs(self):
        # --- 1. センサデータの取得 ---
        accel = self.data.sensor("body_accel").data
        gyro = self.data.sensor("body_gyro").data

        accel_roll = np.arctan2(accel[1], accel[2])
        gyro_roll = gyro[0]
        dt_control = self.dt * 10
        self.filtered_roll = self.alpha * (self.filtered_roll + gyro_roll * dt_control) + (1 - self.alpha) * accel_roll

        gyro_roll_noise_std = 0.1
        # --- 2. 各種パラメータ ---
        roll_rad = self.filtered_roll
        body_roll_vel = gyro[0] + np.random.normal(0, gyro_roll_noise_std) 
        body_yaw_vel = gyro[2] 
        l_wheel_vel = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        r_wheel_vel = -self.data.qvel[self.model.jnt_dofadr[self.r_wheel_id]]

        return np.array([roll_rad, body_roll_vel, body_yaw_vel, l_wheel_vel, r_wheel_vel], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.filtered_roll = 0.0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.data.ctrl[0] = action[0] * 0.021 # wheel_hinge_left
        self.data.ctrl[1] = -action[0] * 0.021 # wheel_hinge_right

        # 10ms ごとに学習
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        l_vel = obs[3]
        r_vel = obs[4]

        # 終了判定 45度(0.78rad)より傾くと終了
        roll = obs[0] 
        terminated = bool(abs(roll) > 0.78)

        # 報酬
        action_penalty = np.sum(np.square(action - self.prev_action))
        reward = float(
            -0.1 * action_penalty # actionの連続値可
            -0.1 * np.sum(np.square(action)) # actionの大きさペナルティ
            # -1.0 * (l_vel**2 + r_vel**2)
            -10.0 * obs[0]**2    # 傾きペナルティ
            -5.0 * obs[1]**2    # 揺れペナルティ
            # -2.0 * obs[2]**2    # その場回転ペナルティ
            # -3.0 * abs(action[0] - action[1])
            # -0.1 * obs[3]**2    # タイヤのスピードペナルティ
            # -0.1 * obs[4]**2    # タイヤのスピードペナルティ
            +10.0 * (abs(obs[0]) < 0.0872) # 倒立報酬(5度以内)
            +1 # 生存報酬
        )
        self.prev_action = action.copy()

        truncated = False     # 時間切れならTrue
        info = {}             # おまけ情報

        return obs, reward, terminated, truncated, info
