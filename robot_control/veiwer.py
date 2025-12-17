# まだ使ってない

import mujoco
import mujoco_viewer
import numpy as np


class Viewer(mujoco_viewer.MujocoViewer):
    def __init__(self, mjcf_path: str, camera_config: dict = None, *args, **kwargs):
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        data = mujoco.MjData(model)

        super().__init__(model=model, data=data, *args, **kwargs)

        # 初期のカメラ位置の変更
        try:
            self.cam.lookat = np.array(camera_config["lookat"])
            self.cam.distance = camera_config["distance"]
            self.cam.azimuth = camera_config["azimuth"]
            self.cam.elevation = camera_config["elevation"]
        except KeyError as e:
            print(e)

    # for _create_overlay
    TOPLEFT = mujoco.mjtGridPos.mjGRID_TOPLEFT
    TOPRIGHT = mujoco.mjtGridPos.mjGRID_TOPRIGHT
    BOTTOMLEFT = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
    BOTTOMRIGHT = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

    def add_overlay(self, gridpos, text1, text2):
        if gridpos not in self._overlay:
            self._overlay[gridpos] = ["", ""]
        self._overlay[gridpos][0] += text1 + "\n"
        self._overlay[gridpos][1] += text2 + "\n"

    def _callback(self):
        pass

    def __call__(self):
        for _ in range(100000):
            self._callback()
            mujoco.mj_step(self.model, self.data)
            self.render()
            if not self.is_alive:
                break

        self.close()