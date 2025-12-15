import mujoco
import mujoco.viewer
import numpy as np
import time

# モデル定義（viewerなし）
model = mujoco.MjModel.from_xml_string("""
<mujoco>
    <option timestep="0.01"/>
    <compiler assetdir="../stl"/>
    
    <asset>
        <mesh name="body" file="body.stl"/>
        <mesh name="tire" file="stl.stl"/>
    </asset>
    
    <worldbody>
        <!-- 照明：上から照らす -->
        <light name="light1" pos="0 0 3" dir="0 0 -1" directional="true" castshadow="false"/>
    
    
        <body name="body" pos="0 0 0.0">
        <joint type="free"/>
        <inertial pos="-0.018 -0.044 61.617" mass="0.049217" diaginertia="3.084924e-5 4.0411796e-5 2.5326781e-5"/>
        <geom type="mesh" mesh="case"rgba="0.4 0.4 0.4 1"/>
        
        <!-- モーター（親） -->
        <body name="motor" pos="0 0 0">
            <inertial mass="0.032212" diaginertia="3.289826e-6 5.548697e-6 3.290785e-6" pos="0 0.009627 0.034116"/>
            <geom type="mesh" mesh="motor" rgba="0.8 0.8 0.85 1"/>
    
            <!-- ホイール（回転する子） -->
            <body name="wheel" pos="0 0 0">
            <!-- Z軸まわりに回転（0 0 1） -->
            <joint name="wheel_hinge" type="hinge" axis="0 1 0" pos="0 0 0.034116" damping="0.00005"/>
            <inertial mass="0.036643" diaginertia="7.825643e-6 1.4733674e-5 7.825642e-6" pos="0 0.004659 0.034116"/>
            <geom type="mesh" mesh="wheel" rgba="0.8 0.8 0.85 1"/>
            </body>
        </body>
        </body>
    
    
        <!-- 地面 -->
        <geom type="plane" size="20 20 0.1" pos="0 0 0" rgba="0.0 0.5 0.0 1"/>
    </worldbody>
    
    <actuator>
        <motor joint="wheel_hinge" ctrlrange="-1.0 1.0" gear="0.0265"/>
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
                                       