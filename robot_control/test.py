import mujoco
import mujoco.viewer
import numpy as np
import time
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
kp = 0.0
ki = 0.0
kd = 0.0
pre_time = 0.0

def get_absolute_roll():
    quat = data.xquat[body_id]
    w, x, y, z = quat
    roll_rad = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    return roll_rad

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():


        # data.ctrl[0] = 1.0 * output_max
        # data.ctrl[1] = 1.0 * output_max


        mujoco.mj_step(model, data)

        # ビューアを更新
        viewer.sync()
        time.sleep(model.opt.timestep)