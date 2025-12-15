import mujoco
import mujoco.viewer
import time
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "pendulum.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        
        # data.ctrl[0] = 5.0 

        mujoco.mj_step(model, data)

        # ビューアを更新
        viewer.sync()
        time.sleep(model.opt.timestep)