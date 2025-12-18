import mujoco
import mujoco.viewer
import time
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "pendulum.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

output_max = 0.021

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        
        data.ctrl[0] = 1.0 * output_max
        data.ctrl[1] = 1.0 * output_max

        mujoco.mj_step(model, data)

        # ビューアを更新
        viewer.sync()
        time.sleep(model.opt.timestep)