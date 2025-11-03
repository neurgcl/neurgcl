import os
import numpy as np

import pybullet as op

from bl_sim_env.utils import np_tf
from bl_sim_env.utils.np_tf import mr
from bl_sim_env.utils.config import PATH_ROOT


class SphereManager:
    def __init__(self, pybullet_client):
        self.p = pybullet_client
        self.spheres = []
        self.color = [0.7, 0.1, 0.1, 1]
        self.color = [0.63, 0.07, 0.185, 1]
        # self.color = [0.8500, 0.3250, 0.0980, 1]

    def create_sphere(self, position, radius, color):
        sphere = self.p.createVisualShape(
            self.p.GEOM_SPHERE, radius=radius, rgbaColor=color, specularColor=[0, 0, 0, 1]
        )
        sphere = self.p.createMultiBody(baseVisualShapeIndex=sphere, basePosition=position)
        self.spheres.append(sphere)

    def initialize_spheres(self, obstacle_array):
        for obstacle in obstacle_array:
            self.create_sphere(obstacle[0:3], obstacle[3], self.color)

    def delete_spheres(self):
        for sphere in self.spheres:
            self.p.removeBody(sphere)
        self.spheres = []

    def update_spheres(self, obstacle_array):
        if (obstacle_array is not None) and (len(self.spheres) == len(obstacle_array)):
            for i, sphere in enumerate(self.spheres):
                self.p.resetBasePositionAndOrientation(sphere, obstacle_array[i, 0:3], [1, 0, 0, 0])
        else:
            print("Number of spheres and obstacles do not match")
            self.delete_spheres()
            self.initialize_spheres(obstacle_array)


class BLPanda:
    def __init__(
        self, bullet_client: op = None, base_pos=[0, 0, 0], base_quat=[0, 0, 0, 1], useFixedBase=True, urdf_path=None
    ) -> None:
        self.p = bullet_client
        self.n_dof = 7
        self.base_pos = np.array(base_pos)
        self.base_quat = np.array(base_quat)
        # self.p.setAdditionalSearchPath('content/assets/urdf/franka_description')

        # urdf_path = urdf_path or PANDA_URDF
        # urdf_path = urdf_path or os.path.join(PATH_ROOT, 'data/urdf/bullet/franka_panda/panda_nofinger_ee.urdf')
        urdf_path = urdf_path or os.path.join(PATH_ROOT, 'data/urdf/bullet/franka_panda/panda_nofinger_tcp_nobase.urdf')


        self.urdf_path = urdf_path
        # flags = op.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        # self.robotId = self.p.loadURDF(urdf_path, self.base_pos, self.base_quat, useFixedBase=True, flags=flags)

        self.jpos_home = np.deg2rad([0.0, -45, 0.0, -135, 0, 90, 45])  # ~ [0.3, 0, 0.6]
        self.useFixedBase = useFixedBase
        # self.T_base2link0 = np_tf.xyzrpydeg2T([0, 0, 0.33, 0, 0, 0])
        self.reset()

    def reset(self):
        self.id_robot = self.p.loadURDF(self.urdf_path, self.base_pos, self.base_quat, useFixedBase=self.useFixedBase)
        index = 0

        self.tcp_joint = 'panda_hand_tcp_joint'
        joint_names = [f'panda_joint{i}' for i in range(1, 1 + self.n_dof + 1)] + [self.tcp_joint]

        self.n_joints = len(joint_names)

        map_j_id = {}
        for j in range(self.p.getNumJoints(self.id_robot)):
            self.p.changeDynamics(self.id_robot, j, linearDamping=0, angularDamping=0)
            info = self.p.getJointInfo(self.id_robot, j)
            jointName = info[1]
            jointType = info[2]
            map_j_id[jointName.decode()] = j
        self.map_j_id = map_j_id

        act_joint_names = [f'panda_joint{i}' for i in range(1, 1 + self.n_dof)]
        self.act_joint_names = act_joint_names
        self.id_act_joints = [map_j_id[jname] for jname in act_joint_names]

        self.id_tcp = self.map_j_id[self.tcp_joint]
        # self.id_hand = self.map_j_id['panda_hand_tcp_joint']
        self.id_ee = self.id_tcp
        # self.id_ee = self.id_tcp
        # T_w2tcp = self.get_T_robot2joint(self.id_tcp)
        # T_w2hand = self.get_T_robot2joint(self.id_hand)
        # self.T_hand2tcp = mr.TransInv(T_w2hand) @ T_w2tcp
        # self.T_hand2tcp = np_tf.xyzrpydeg2T([0, 0, 0.1034, 0, 0, 0])
        self.T_hand2tcp = np_tf.xyzrpydeg2T([0, 0, 0, 0, 0, 0])

        self.reset_q(self.jpos_home)

        # ee_t, ee_quat = self.get_ee_pose()
        # shape_indicator = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01)
        # self.id_init = p.createMultiBody(
        #     baseMass=0,
        #     baseVisualShapeIndex=shape_indicator,
        #     basePosition=ee_t,
        # )
        # p.changeVisualShape(self.id_init, -1, rgbaColor=[1.0, 1.0, 0, 0.5])

        # self.id_target = p.createMultiBody(
        #     baseMass=0,
        #     baseVisualShapeIndex=shape_indicator,
        #     basePosition=ee_t,
        # )
        # p.changeVisualShape(self.id_target, -1, rgbaColor=[1.0, 0, 1, 0.5])
        # self.axiscreator(self.id_init)
        # self.axiscreator(self.id_target)

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
            parentLinkIndex=linkId,
        )

        y_axis = self.p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0.1, 0],
            lineColorRGB=[0, 1, 0],
            lineWidth=lineWidth,
            lifeTime=0,
            parentObjectUniqueId=bodyId,
            parentLinkIndex=linkId,
        )

        z_axis = self.p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, 0.1],
            lineColorRGB=[0, 0, 1],
            lineWidth=lineWidth,
            lifeTime=0,
            parentObjectUniqueId=bodyId,
            parentLinkIndex=linkId,
        )
        return [x_axis, y_axis, z_axis]

    def reset_q(self, q, qd=None):
        """
        Force reset the robot to a specific joint configuration
        """
        if qd is not None:
            self.p.resetJointStatesMultiDof(self.id_robot, self.id_act_joints, q[:, np.newaxis], qd[:, np.newaxis])
        else:
            self.p.resetJointStatesMultiDof(self.id_robot, self.id_act_joints, q[:, np.newaxis])
            
    def send_q_cmd(self, q):
        return self.reset_q(q)

    def get_ee_pose(self, ee_joint_id=None):
        if ee_joint_id is None:
            ee_joint_id = self.id_ee
        state = self.p.getLinkState(self.id_robot, ee_joint_id)
        ee_pos = state[4]
        ee_quat = state[5]
        return ee_pos, ee_quat

    def get_T_robot2joint(self, joint_id):
        pq_base2joint = self.get_ee_pose(joint_id)
        T_base2joint = np_tf.xyzquat2T(pq_base2joint[0], pq_base2joint[1])
        return T_base2joint
    
    def get_T_w2ee(self, q=None):
        """
        Note: This may reset_q
        """
        if q is not None:
            self.reset_q(q)
        return self.get_T_robot2joint(self.id_ee)
    
    def get_ik(self, T_goal, maxIter=2000):
        pose_goal = np_tf.T2xyzquat(T_goal)
        q_goal = self.p.calculateInverseKinematics(
            self.id_robot, self.id_ee, pose_goal[0], pose_goal[1], maxNumIterations=maxIter
        )
        q_goal = np.array(q_goal)
        return q_goal
        
    def get_ik_checkerr(self, T_target, maxIter=2000, err_t_max = 1e-3, err_r_max = 0.05, verbose=False, ret_info=False):
        """
        Note: This may reset_q
        """
        q_target = self.get_ik(T_target, maxIter)
        T_target1 = self.get_T_w2ee(q_target)

        T_diff = np.linalg.inv(T_target) @  T_target1
        err = np_tf.T2tw(T_diff)
        err_t, err_r = err[:3], err[3:]
        err_t = np.linalg.norm(err_t)
        err_r = np.linalg.norm(err_r)
        err_r = np.rad2deg(err_r)

        info={'err_t': err_t, 'err_r': err_r}
        if verbose:
            print(f"err_t: {err_t}, err_r: {err_r}")

        if err_t < err_t_max and err_r < err_r_max:
            pass
        else:
            q_target=None
   
        if ret_info:
            return q_target, info
        else:
            return q_target

        
    def set_joint_target_positions(self, q):
        self.p.setJointMotorControlArray(
            self.id_robot, self.id_act_joints, self.p.POSITION_CONTROL, q, forces=[5 * 240.0] * self.n_dof
        )

        # for i in range(self.n_dof):
        #     self.p.setJointMotorControl2(self.id_robot, i, self.p.POSITION_CONTROL, qs[i], force=5 * 240.0)
        # self.set_finger_positions(0.04)

    # def set_finger_positions(self, gripper_opening):
    #     self.p.setJointMotorControl2(self.id_robot, 9, self.p.POSITION_CONTROL, gripper_opening / 2, force=5 * 240.0)
    #     self.p.setJointMotorControl2(self.id_robot, 10, self.p.POSITION_CONTROL, -gripper_opening / 2, force=5 * 240.0)

    def get_joint_positions(self):
        joint_state = []
        for i in range(self.n_dof):
            joint_state.append(self.p.getJointState(self.id_robot, i)[0])
        return joint_state

    def get_joint_dstates(self, env_handle=None, robot_handle=None):
        l_jpos = self.p.getJointStates(self.id_robot, self.id_act_joints)

        # reformat state to be similar ros jointstate:
        joint_state = {'name': self.act_joint_names, 'position': [], 'velocity': [], 'acceleration': []}

        for i in range(len(l_jpos)):
            joint_state['position'].append(l_jpos[i][0])
            joint_state['velocity'].append(l_jpos[i][1])
        joint_state['position'] = np.ravel(joint_state['position'])
        joint_state['velocity'] = np.ravel(joint_state['velocity'])
        joint_state['acceleration'] = np.zeros_like(joint_state['velocity'])

        return joint_state

    def command_robot_position(self, q):
        self.set_joint_target_positions(q)

    def set_robot_state(self, q, qd):
        self.reset_q(q, qd)
