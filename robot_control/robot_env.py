import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import random
from collections import deque

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
        self.odom = np.zeros(2)
        self.latency_step = 3
        # 指令値の履歴キュー
        self.control_queue = deque([np.zeros(2)] * (self.latency_step + 1), maxlen=self.latency_step+1)
        # エンコーダの履歴キュー
        self.encoder_queue = deque([np.zeros(2)] * (self.latency_step + 1), maxlen=self.latency_step+1)

        # --- 5. 報酬設定用変数の初期化 ---
        self.pre_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _get_robot_angle(self):
        # センサデータの取得
        accel = self.data.sensor("body_accel").data
        gyro = self.data.sensor("body_gyro").data
        # 相補フィルタでroll推定
        accel_roll_noise_std = 0.00015
        accel_roll = np.arctan2(accel[1], accel[2]) + np.random.normal(0, accel_roll_noise_std)
        gyro_roll_noise_std = 0.0008
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
        self.encoder_queue.append(np.array([l_wheel_vel, r_wheel_vel]))
        delay_wheel_vel = self.encoder_queue[0]

        forward_input = 0.0

        obs = np.array([roll_rad, gyro_rad, delay_wheel_vel[0], delay_wheel_vel[1], forward_input], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.control_queue = deque([np.zeros(2)] * self.control_queue.maxlen, maxlen=self.control_queue.maxlen)
        self.encoder_queue = deque([np.zeros(2)] * self.control_queue.maxlen, maxlen=self.control_queue.maxlen)

        # 角度のランダム化
        random_roll = self.np_random.uniform(low=-0.15, high=0.15)
        quat = np.array([np.cos(random_roll/2), np.sin(random_roll/2), 0, 0])
        self.data.qpos[3:7] = quat
        self.data.qpos[0:3] = [0, 0, 0.012]
        self.data.qpos[7:] = 0.0

        # 2. 角速度のランダム化
        # low, high の値は必要に応じて調整（単位: rad/s）
        random_roll_vel = self.np_random.uniform(low=-3.0, high=3.0)
        self.data.qvel[3] = random_roll_vel  # X軸周りの角速度 (Roll velocity)

        # 状態の更新
        mujoco.mj_forward(self.model,self.data)

        # 変数初期化
        self.filtered_roll = 0.0
        self.odom = np.zeros(2)
        self.pre_action = np.zeros(self.action_space.shape, dtype=np.float32)
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # action_std_noise = 0.0233 * 0.002
        # base_torque = action[0]*0.0233 + np.random.normal(0, action_std_noise)
        # bias= 0.0021

        # if base_torque > bias:
        #     torque_l = base_torque-bias
        #     torque_r = -(base_torque-bias)
        # elif base_torque < -bias:
        #     torque_l = base_torque+bias
        #     torque_r = -(base_torque+bias)
        # else:
        #     torque_l = 0.0   
        #     torque_r = 0.0   

        action_std_noise = 0.0212 * 0.002
        base_torque = action[0]*0.0212 + np.random.normal(0, action_std_noise)
        torque_l = base_torque
        torque_r = -base_torque
        
        self.control_queue.append(np.array([torque_l, torque_r]))
    
        
        delayed_ctrl = self.control_queue[0]
        # self.data.ctrl[0] = delayed_ctrl[0] + np.random.normal(0, action_std_noise)
        # self.data.ctrl[1] = delayed_ctrl[1] + np.random.normal(0, action_std_noise)
        self.data.ctrl[0] = delayed_ctrl[0]
        self.data.ctrl[1] = delayed_ctrl[1]


        # 制御遅延           
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)           

        obs = self._get_obs()
        error = obs[4] - obs[0]

        self.odom[0] += 0.03 * obs[2] * self.dt # dL = rωdt
        self.odom[1] += 0.03 * obs[3] * self.dt # dL = rωdt
        wheel_odom = np.linalg.norm(self.odom)

        # 報酬
        action_penalty = np.sum(np.square(action - self.pre_action))
        reward = float(
            # -0.1 * action**2 # アクションの大きさ
            -0.01 * action_penalty # actionの滑らかさ
            -2.0 * obs[1]**2 # 角速度ペナルティ
            -0.01 * obs[2]**2 # タイヤ速度ペナルティ
            -15.0 * self.odom[0]**2 # 移動距離ペナルティ
            +10.0 * (2 - abs(error))
        )
        self.pre_action = action.copy()

        # 終了判定 45度(0.78rad)より傾くと終了
        roll = obs[0] 
        terminated = bool(abs(roll) > 0.785)

        truncated = False     # 時間切れならTrue
        info = {}   # おまけ情報

        return obs, reward, terminated, truncated, info
