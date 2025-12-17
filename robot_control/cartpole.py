import mujoco
import mujoco.viewer
import numpy as np
import time
 
# モデル定義（viewerなし）
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
 
  <worldbody>
    <body name="origin" pos="0 0 0">
      <!-- Y軸（緑） -->
      <geom name="y_axis" type="capsule" fromto="0 0 0 0 -0.5 0" size="0.01" rgba="0 1 0 1"/>
    </body>
 
    <body name="base" pos="0 0 0">
      <geom type="box" size="0.2 0.2 0.02" rgba="0.6 0.6 0.6 1"/>
      <body name="pendulum" pos="0 0 0.02">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-90 90" limited="true" damping="1.0"/>
        <inertial pos="0 0 0.4" mass="1.0" diaginertia="0.1 0.1 0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.8" size="0.02" rgba="0.2 0.6 0.8 1"/>
      </body>
    </body>
  </worldbody>
 
  <actuator>
    <motor joint="hinge" ctrlrange="-10 10" gear="1"/>
  </actuator>
</mujoco>
""")
 
data = mujoco.MjData(model)
data.qpos[0] = np.deg2rad(20)  # 初期角度20度
 
# ゲイン
Kp = 13.0
Kd = 5.0
 
 
# Viewerつきループ
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        theta = data.qpos[0]
        omega = data.qvel[0]
        torque = -Kp * theta - Kd * omega
 
        disturbance = np.random.normal(loc=0.0, scale=1.0)
        data.ctrl[0] = torque + disturbance
 
        mujoco.mj_step(model, data)
 
        viewer.sync()                # ステップ後の状態をviewerに反映
        time.sleep(model.opt.timestep)  # 描画ループ安定化