import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import os

# 倒立振子xmlのインポート
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "pendulum.xml")

# mujocoモデルとbase_id
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

# pidパラメータ
output_max = 0.021
target_rad = 0.0
kp = 9.8
ki = 0.0
kd = 1.0
pre_time = 0.0
pre_error = 0.0
integral = 0.0

def get_absolute_roll():
    quat = data.xquat[body_id]
    w, x, y, z = quat
    roll_rad = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    return roll_rad

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # タイマー
        now = time.perf_counter()
        
        # pidコントローラ
        roll = get_absolute_roll()
        # dt = (now - pre_time) * 1000 # ミリ秒
        dt = model.opt.timestep # タイムスリープで計算
        pre_time = now
        error = target_rad - roll
        integral += error * dt
        deriv = (error - pre_error) / dt
        pre_error = error

        output = kp * error + ki * integral + kd * deriv + random.uniform(-0.6, 0.6)
        output = np.clip(output, -1.0, 1.0)

        data.ctrl[0] = - output * output_max
        data.ctrl[1] = output * output_max

        print(output)

        mujoco.mj_step(model, data)

        # ビューアを更新
        viewer.sync()
        time.sleep(model.opt.timestep)