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
        self.dt = self.model.opt.timestep * 10 # 10ステップ分
        self.filtered_roll = 0.0
        self.alpha = 0.98

        # --- 5. 報酬設定用変数の初期化 ---
        self.pre_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _get_robot_angle(self):
        # センサデータの取得
        accel = self.data.sensor("body_accel").data
        gyro = self.data.sensor("body_gyro").data
        # 相補フィルタでroll推定
        accel_roll = np.arctan2(accel[1], accel[2])
        gyro_roll_noise_std = 0.0006
        gyro_roll = gyro[0] + np.random.normal(0, gyro_roll_noise_std)
        self.filtered_roll = self.alpha * (self.filtered_roll + gyro_roll * self.dt) + (1 - self.alpha) * accel_roll

        return self.filtered_roll

    def _get_obs(self):
        # センサデータの取得
        accel = self.data.sensor("body_accel").data
        gyro = self.data.sensor("body_gyro").data

        # 観測空間
        roll_rad = self._get_robot_angle()
        gyro_rad = gyro[0]
        l_wheel_vel = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        r_wheel_vel = -self.data.qvel[self.model.jnt_dofadr[self.r_wheel_id]]
        forward_input = 0.0

        return np.array([roll_rad, gyro_rad, l_wheel_vel, r_wheel_vel, forward_input], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        random_roll = self.np_random.uniform(low=-0.0, high=0.0)
        quat = np.array([np.cos(random_roll/2), np.sin(random_roll/2), 0, 0])
        self.data.qpos[3:7] = quat

        self.data.qpos[0:3] = [0, 0, 0.012]
        self.data.qpos[7:] = 0.0

        mujoco.mj_forward(self.model,self.data)

        # 変数初期化
        self.filtered_roll = 0.0
        self.pre_action = np.zeros(self.action_space.shape, dtype=np.float32)
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.data.ctrl[0] = action * 0.0276
        self.data.ctrl[1] = -action * 0.0276

        # 10ms ごとに学習
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)           

        obs = self._get_obs()
        error = obs[4] - obs[0]

        # 報酬
        action_penalty = np.sum(np.square(action - self.pre_action))
        reward = float(
            -0.1 * action**2 # アクションの大きさ
            -0.1 * action_penalty # actionの滑らかさ
            -0.5 * obs[1]**2 # 角速度ペナルティ
            -0.01 * obs[2]**2 # タイヤ速度ペナルティ
            -0.01 * obs[3]**2 # タイヤ速度ペナルティ
            +5.0 * (2 - abs(error)) # 角度報酬
        )
        self.pre_action = action.copy()

        # 終了判定 45度(0.78rad)より傾くと終了
        roll = obs[0] 
        terminated = bool(abs(roll) > 0.785)

        truncated = False     # 時間切れならTrue
        info = {}             # おまけ情報

        return obs, reward, terminated, truncated, info
