import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import os
from PID_for_PPO import PID_controller
import socket

teleplotAddr = ("127.0.0.1",47269)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def sendTelemetry(name, value):
    now = time.time() * 1000
    msg = name+":"+str(now)+":"+str(value)+"|g"
    sock.sendto(msg.encode(), teleplotAddr)

# 倒立振子xmlのインポート
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "pendulum.xml")

# mujocoモデルとbase_id
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

# pidパラメータ
speed_max = 0.0276
target_rad = 0.0
kp = 1
ki = 0.0
kd = 0.05
pre_time = 0.0
pre_error = 0.0
integral = 0.0
dt = 0.0
filtered_roll = 0.0

pid_forward = PID_controller(kp, ki, kd)

def get_absolute_roll():
    quat = data.xquat[body_id]
    w, x, y, z = quat
    roll_rad = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    roll_rad += np.random.normal(0, 0.003)
    # roll_deg = roll_rad * 180 / np.pi
    return roll_rad

def _get_robot_angle(dt):
    global filtered_roll
    # センサデータの取得
    accel = data.sensor("body_accel").data
    gyro = data.sensor("body_gyro").data
    # 相補フィルタでroll推定
    accel_roll = np.arctan2(accel[1], accel[2])
    gyro_roll_noise_std = 0.01
    gyro_roll = gyro[0] + np.random.normal(0, gyro_roll_noise_std)
    filtered_roll = 0.8 * (filtered_roll + gyro_roll * dt) + (1 - 0.8) * accel_roll
    filtered_deg = filtered_roll * 180 / np.pi

    return filtered_deg

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # タイマー
        now = time.perf_counter()
        
        # pidコントローラ
        roll = get_absolute_roll()
        # dt = (now - pre_time) * 1000 # ミリ秒
        dt = model.opt.timestep * 10 # タイムスリープで計算

        # roll = _get_robot_angle(dt)

        pre_time = now
        error = target_rad - roll
        integral += error * dt
        deriv = (error - pre_error) / dt
        pre_error = error

        # # output = kp * error + ki * integral + kd * deriv + random.uniform(-1.0, 1.0)
        output = kp * error + ki * integral + kd * deriv
        output = np.clip(output, -1.0, 1.0)

        # output = pid_forward.calc(0, roll, dt)

        data.ctrl[0] = -output * speed_max
        data.ctrl[1] = output * speed_max

        sendTelemetry("roll", roll)
        sendTelemetry("deriv", deriv)
        sendTelemetry("ouput", output)
        print(abs(output))

        for _ in range(10):
            mujoco.mj_step(model, data)

        # ビューアを更新
        viewer.sync()
        time.sleep(model.opt.timestep * 10)