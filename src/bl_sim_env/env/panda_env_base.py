import time

import numpy as np
import pybullet as op
import pybullet_data
import trimesh
from bl_sim_env.env.bl_extras_base import BLPanda
from bl_sim_env.robot.franka_dh_fk import fk_tfs
from bl_sim_env.utils import np_tf
from bl_sim_env.utils.config import PATH_ROOT
from pybullet_utils.bullet_client import BulletClient
from trimesh.transformations import scale_matrix


class PandaEnv:
    def __init__(
        self, useGUI=True, plane_pos=np.array([0, 0, 0]), flag_show_txt=True
    ) -> None:
        self._useGUI = useGUI
        self.useRealTimeSimulation = False
        self.t = 0
        if self._useGUI:
            self.p = BulletClient(op.GUI, options="--mp4fps=10")
            # self.p = BulletClient(op.GUI)
        else:
            self.p = BulletClient(op.DIRECT)

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        self.pos_debug_txt = [0, -0.5, 0]

        # axis creator
        self.id_init = None
        self.id_target = None
        self.plane_pos = plane_pos
        self.flag_show_txt = flag_show_txt

        self.reset()
        self.reset_camera()

    def reset(self):
        p = self.p
        p.resetSimulation()

        # if not self._useGUI:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # only show main window
        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.setGravity(0, 0, -10)
        self.id_plane = p.loadURDF("plane.urdf", self.plane_pos)

        self.robot = BLPanda(
            p,
            np.array([0, 0, 0.0]),
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        self.view_dict = dict(
            cameraTargetPosition=[0, 0, 0.5],
            distance=2,
            yaw=60,
            pitch=-20,
            roll=0,
            upAxisIndex=2,
        )

        self.col_objs = {}
        self.logId = None
        self.txt_item = None

    def reset_camera(
        self, distance=1.2, yaw=-45, pitch=-30, cameraTargetPosition=[0.42, -0.21, 0.36]
    ):
        env = self
        env.view_dict["cameraTargetPosition"] = cameraTargetPosition
        env.view_dict["distance"] = distance
        env.view_dict["roll"] = 0
        env.view_dict["pitch"] = pitch
        env.view_dict["yaw"] = yaw
        env.p.resetDebugVisualizerCamera(
            env.view_dict["distance"],
            env.view_dict["yaw"],
            env.view_dict["pitch"],
            env.view_dict["cameraTargetPosition"],
        )

    def update_txt(self, txt, color=[0, 1, 0]):
        if not self.flag_show_txt:
            return
        p = self.p
        if self.txt_item is not None:
            p.removeUserDebugItem(self.txt_item)
            self.txt_item = None
        # color = [0, 1, 0]
        self.txt_item = p.addUserDebugText(
            txt, self.pos_debug_txt, textColorRGB=color, textSize=5
        )

    def remove_txt(self):
        if self.txt_item is not None:
            self.p.removeUserDebugItem(self.txt_item)
            self.txt_item = None

    def update_txt_step_dis(self, i, d_bl=None, d_target=0.1, d_warn=0.09, prefix=""):
        if d_bl is None:
            txt = f"[{i:03d}]{prefix}"
        else:
            txt = f"[{i:03d}]{prefix} d:{d_bl:.4f}"
        if d_bl < d_warn:
            color = [1, 0, 0]
        elif d_bl < d_target:
            color = [1, 1, 0]
        else:
            color = [0, 1, 0]
        self.update_txt(txt, color=color)

    def __del__(self):
        if self.p is not None:
            self.p.disconnect()
            self.p = None

    def step(self):
        if self.useRealTimeSimulation:
            self.t = (
                time.time()
            )  # (dt, micro) = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f').split('.')
            # t = (dt.second/60.)*2.*math.pi
        else:
            self.t = self.t + 0.001

        self.p.stepSimulation()

    def reset_obj_pose(self, obj_id, pos, quat=op.getQuaternionFromEuler([0, 0, 0])):
        self.p.resetBasePositionAndOrientation(obj_id, pos, quat)

    def start_record(self, logName="test.mp4"):
        self.stop_record()
        self.logId = self.p.startStateLogging(op.STATE_LOGGING_VIDEO_MP4, logName)
        self.last_log_name = logName

    def stop_record(self):
        if self.logId is not None:
            self.p.stopStateLogging(self.logId)
            self.logId = None

    def axiscreator(self, bodyId, linkId=-1):
        # print(f'axis creator at bodyId = {bodyId} and linkId = {linkId} as XYZ->RGB')
        lineWidth = 0.5
        x_axis = self.p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0.1, 0, 0],
            lineColorRGB=[1, 0, 0],
            lineWidth=lineWidth,
            lifeTime=0,
            parentObjectUniqueId=bodyId,
            # parentLinkIndex = linkId,
        )

        y_axis = self.p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0.1, 0],
            lineColorRGB=[0, 1, 0],
            lineWidth=lineWidth,
            lifeTime=0,
            parentObjectUniqueId=bodyId,
            # parentLinkIndex = linkId,
        )

        z_axis = self.p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, 0.1],
            lineColorRGB=[0, 0, 1],
            lineWidth=lineWidth,
            lifeTime=0,
            parentObjectUniqueId=bodyId,
            # parentLinkIndex = linkId,
        )
        return [x_axis, y_axis, z_axis]

    def reset_init_target(self, T_target=None, T_init=None, q_0=None):
        p = self.p

        if q_0 is not None:
            self.robot.reset_q(q_0)

        if T_target is None:
            T_target = fk_tfs(q_0)[-1]

        T_hand2tcp = self.robot.T_hand2tcp

        pq_ee_cur = self.robot.get_ee_pose()

        T_init = np_tf.xyzquat2T(*pq_ee_cur) @ T_hand2tcp
        pq_init = np_tf.T2xyzquat(T_init)
        T_target = T_target @ T_hand2tcp
        pq_target = np_tf.T2xyzquat(T_target)

        if self.id_init is None:
            shape_indicator = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01)
            self.id_init = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=shape_indicator,
                basePosition=pq_init[0],
                baseOrientation=pq_init[1],
            )
            p.changeVisualShape(self.id_init, -1, rgbaColor=[1.0, 1.0, 0, 0.5])

            self.id_target = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=shape_indicator,
                basePosition=pq_target[0],
                baseOrientation=pq_target[1],
            )
            p.changeVisualShape(self.id_target, -1, rgbaColor=[1.0, 0, 1, 0.5])

            self.axiscreator(self.id_init)
            self.axiscreator(self.id_target)

        self.reset_obj_pose(self.id_init, *pq_init)
        self.reset_obj_pose(self.id_target, *pq_target)

 
if __name__ == "__main__":
    obj_path = PATH_ROOT + "/data/obj_meshs/bun_zipper.obj"
    mesh = trimesh.load(obj_path)
    scale = scale_matrix(1.2)
    mesh.apply_transform(scale)

    pts_raw, _ = trimesh.sample.sample_surface_even(mesh, 1000)
    # pts=deepcopy(pts_raw)

    T_w2bunny = np_tf.xyzrpy2T([0.3, 0, 0.45, 0, 0, 0], degrees=True)
    pts = np_tf.tf_pts(T_w2bunny, pts_raw)

    env = PandaEnv()
    p = env.p
    robot = env.robot

    # fmt: off
    q_0 = np.array([-0.0965195386873646,-0.48818972252726783,-0.5262399291814241,-2.0770811368204902,-0.23848617636484631,1.6444697759066003,0.23276192924707884])
    q_target = np.array([0.09651953869727141, -0.48818972254287984, 0.5262399291691557, -2.0770811368275495,0.23848617638248512, 1.6444697759146427, 1.3380343975414601])
    # fmt: on

    T_0 = fk_tfs(q_0)[-1]
    T_target = fk_tfs(q_target)[-1]

    robot.reset_q(q_0)
    env.reset_col_obj(T_w2bunny)

    # p.setAdditionalSearchPath(PATH_ROOT+'/data/obj_meshs')
    # col_bunny_id =p.createCollisionShape(p.GEOM_MESH,
    #                        fileName=PATH_ROOT+"/data/obj_meshs/bun_zipper.obj",
    #                        flags=p.GEOM_FORCE_CONCAVE_TRIMESH|p.GEOM_CONCAVE_INTERNAL_EDGE,
    #                        meshScale=[1.0, 1.0, 1.0])
    # col_bunny_id = p.createCollisionShape(
    #     p.GEOM_MESH,
    #     vertices=mesh.vertices,
    #     indices=np.array(mesh.faces).flatten(),
    #     # flags=p.GEOM_FORCE_CONCAVE_TRIMESH|p.GEOM_CONCAVE_INTERNAL_EDGE,
    #     meshScale=[1.0, 1.0, 1.0],
    # )
    # bunny_id = p.createMultiBody(0, col_bunny_id, -1, *np_tf.T2xyzquat(T_w2bunny))

    env.view_dict["cameraTargetPosition"] = [0.42, -0.21, 0.36]
    env.view_dict["distance"] = 1.2
    env.view_dict["roll"] = 0
    env.view_dict["pitch"] = -30
    env.view_dict["yaw"] = -45
    p.resetDebugVisualizerCamera(
        env.view_dict["distance"],
        env.view_dict["yaw"],
        env.view_dict["pitch"],
        env.view_dict["cameraTargetPosition"],
    )

    env.step()
    # env.reset_obj_pose(bunny_id, [0.3, 0, 0.45],p.getQuaternionFromEuler([np.pi/2, 0, 0]))
    while True:
        env.step()
        time.sleep(0.01)
