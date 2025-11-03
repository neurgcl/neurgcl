import os

import numpy as np
import torch
import trimesh
from curobo.cuda_robot_model.cuda_robot_model import (CudaRobotModel,
                                                      CudaRobotModelConfig)
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util.logger import log_error

from neurg.my_utils.config import PATH_ROOT


def cal_J_ee_wrt_q_raw(T_w2links):
    T_w2ee = T_w2links[-1]
    point_t = T_w2ee[:3, 3]

    J_ee_wrt_q = torch.empty((6, 7), device=T_w2links.device)

    for i in range(7):
        joint_axis = T_w2links[1 + i][:3, 2]
        # J_x = joint_axis x( point_t - joint_t)
        J_ee_wrt_q[:3, i] = torch.linalg.cross(joint_axis, point_t - T_w2links[1 + i][:3, 3])
        # J_w = joint_axis
        J_ee_wrt_q[3:, i] = joint_axis
    return J_ee_wrt_q


@torch.jit.script
def cal_J_ee_wrt_q(T_w2links):
    T_w2ee = T_w2links[-1]
    point_t = T_w2ee[:3, 3]

    J_ee_wrt_q = torch.empty((6, 7), device=T_w2links.device)

    l_translation = (point_t - T_w2links[1:8, :3, 3]).T
    l_joint_axis = T_w2links[1:8, :3, 2].T

    J_ee_wrt_q[:3] = torch.linalg.cross(l_joint_axis, l_translation, dim=0)
    J_ee_wrt_q[3:] = l_joint_axis
    return J_ee_wrt_q


class MyCudaRobotModel(CudaRobotModel):
    def __init__(self, config: CudaRobotModelConfig):
        super().__init__(config)
        kin_cfg = self.kinematics_config
        self.tor_link_map = kin_cfg.store_link_map.to(torch.int)  # all links in 'link_names'
        self.ret_mesh_link_idxs = None
        if kin_cfg.mesh_link_names is not None:
            mesh_link_idxs = [kin_cfg.link_name_to_idx_map[l] for l in kin_cfg.mesh_link_names]
            mesh_link_idxs = torch.tensor(mesh_link_idxs, device=self.tensor_args.device, dtype=torch.int)
            l_idx = self.tor_link_map.tolist()
            ret_mesh_link_idxs = [l_idx.index(idx) for idx in mesh_link_idxs]
            self.ret_mesh_link_idxs = torch.tensor(ret_mesh_link_idxs, device=self.tensor_args.device, dtype=torch.int)
        else:
            mesh_link_idxs = self.tor_link_map

        self._global_mesh_link_idxs = mesh_link_idxs

    def update_batch_size(self, batch_size, force_update=False, reset_buffers=False):
        if batch_size == 0:
            log_error("batch size is zero")
        if force_update and self._batch_size == batch_size and self.compute_jacobian:
            self.lin_jac = self.lin_jac.detach()  # .requires_grad_(True)
            self.ang_jac = self.ang_jac.detach()  # .requires_grad_(True)
        elif self._batch_size != batch_size or reset_buffers:
            self._batch_size = batch_size
            n_link_name = len(self.link_names)
            self._link_pos_seq = torch.zeros(
                (self._batch_size, n_link_name, 3),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._link_quat_seq = torch.zeros(
                (self._batch_size, n_link_name, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._link_rot_seq = torch.zeros(
                (self._batch_size, n_link_name, 3, 3),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._batch_robot_spheres = torch.zeros(
                (self._batch_size, self.kinematics_config.total_spheres, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.collision_geometry_dtype,
            )
            self._grad_out_q = torch.zeros(
                (self._batch_size, self.get_dof()),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._global_cumul_mat = torch.zeros(
                (self._batch_size, self.kinematics_config.link_map.shape[0], 4, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._col_link_mat = torch.zeros(
                (self._batch_size, n_link_name, 4, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            if self.compute_jacobian:
                self.lin_jac = torch.zeros(
                    [batch_size, 3, self.kinematics_config.n_dofs],
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )
                self.ang_jac = torch.zeros(
                    [batch_size, 3, self.kinematics_config.n_dofs],
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )

    def fk_link_pos_quat(self, q):
        """
        Args:
            q: [batch_size, dof]
        Returns:
            pos: [batch_size, n_links, 3]
            rot: [batch_size, n_links, 4]
        """
        if len(q.shape) > 2:
            raise ValueError("q shape should be [batch_size, dof]")
        batch_size = q.shape[0]
        self.update_batch_size(batch_size, force_update=q.requires_grad)
        link_pos, link_quat, _ = self._cuda_forward(q)
        return link_pos, link_quat

    def fk_link_T44(self, q):
        """
        Args:
            q: [batch_size, dof]
        Returns:
            T_w2links: [batch_size, n_links, 4, 4]
        """
        self.fk_link_pos_quat(q.contiguous())
        torch.index_select(self._global_cumul_mat, 1, self.tor_link_map, out=self._col_link_mat)
        return self._col_link_mat


    def fk_links_ee(self, q):
        """
        Args:
            q: [(n_q, ) dof]
        Returns:
            T_w2links: [(n_q, ) n_links, 4, 4]
            T_w2ee: [(n_q, ) 4, 4]
        """
        single_q = len(q.shape) == 1
        if single_q:
            q = q.unsqueeze(0)

        self.fk_link_pos_quat(q.contiguous())
        torch.index_select(self._global_cumul_mat, 1, self.tor_link_map, out=self._col_link_mat)

        T_w2links = torch.index_select(self._global_cumul_mat, 1, self._global_mesh_link_idxs)

        if self._global_mesh_link_idxs is None:
            ee_idx = -1
        else:
            ee_idx = self.kinematics_config.ee_idx
        T_w2ee = self._col_link_mat[:, ee_idx].clone()

        if single_q:
            T_w2links = T_w2links.squeeze(0)
            T_w2ee = T_w2ee.squeeze(0)
        return T_w2links, T_w2ee

    def np_fk_links_ee(self, q):
        if isinstance(q, np.ndarray):
            q = torch.tensor(q, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        ret = self.fk_links_ee(q)
        return (k.cpu().numpy() for k in ret)

    def fk1q_col_ee(self, q):
        """
        Args:
            q: [dof]
        Returns:
            T_w2cols: [4, 4]
            T_w2ee: [4, 4]
        """
        n_col_link = 8
        T_w2links = self.fk_link_T44(q.unsqueeze(0)).squeeze(0)
        T_w2cols = T_w2links[:, :n_col_link].contiguous()
        return T_w2cols, T_w2links[-1]

    def qp_fk_T44s(self, q):
        """
        Args:
            q: [dof=7]
        Returns:
            T_w2links: [n_links, 4, 4]
        """
        T_w2links = self.fk_link_T44(q.unsqueeze(0))[0].detach().clone()  # (n_links, 4, 4)
        return T_w2links

    def qp_fk_T34s(self, q):
        T_w2links = self.qp_fk_T44s(q)[
            :,
            :3,
        ].contiguous()
        return T_w2links

    def qp_fk_T44s_J(self, q):
        """
        Args:
            q: [dof=7]
        Returns:
            T_w2ee: [4,4]
            J_ee_wrt_q: [6, dof]
            T_wlinks: [n_links, 4, 4]
        """
        # T_w2links = self.qp_fk_T_w2links(q)[:,:3,].contiguous()
        T_w2links = self.qp_fk_T44s(q)

        # J_ee_wrt_q ----------------------------------------------
        # T_w2ee = T_w2links[-1]

        J_ee_wrt_q = cal_J_ee_wrt_q(T_w2links)

        return (T_w2links, J_ee_wrt_q)
    
    def fk_T_w2links(self, qs):
        """
        Args:
            q: [n_q, dof=7]
        Returns:
            T_w2links: [n_q, n(mesh_links), 4, 4]
        """
        # assert len(q.shape) == 1, "q should be a 1D tensor"
        self.fk_link_pos_quat(qs.contiguous())
        T_w2links = torch.index_select(self._global_cumul_mat, 1, self._global_mesh_link_idxs)
        return T_w2links


def get_fk_model(ee_link='panda_hand_tcp', urdf_file=PATH_ROOT + '/data/urdf/bullet/franka_panda/panda_nofinger_tcp.urdf') -> MyCudaRobotModel:
    
    # fmt: off
    base_link = 'panda_link0'

    link_names = [
        "panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", 
        "panda_link5", "panda_link6", "panda_link7", "panda_hand", 
    ]
    mesh_link_names=[
        "panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", 
        "panda_link5", "panda_link6", "panda_link7", "panda_hand"
    ]

    if ee_link != "panda_hand":
        link_names.append(ee_link)
    # fmt: on

    robot_data = {
        'kinematics': {
            "urdf_path": urdf_file,
            'base_link': base_link,
            'ee_link': ee_link,
            'link_names': link_names,
            'mesh_link_names': mesh_link_names,
            'use_global_cumul': True,  # true: copy T44 to self._global_cumul_mat
        },
    }
    robot_cfg = RobotConfig.from_dict(robot_data)
    robot_cfg.kinematics.kinematics_config.link_names
    robot_cfg.kinematics.kinematics_config.fixed_transforms.shape
    
    kin_model = MyCudaRobotModel(robot_cfg.kinematics)
    return kin_model



def update_get_col_mesh_paths_from_robot_model(kin_model: MyCudaRobotModel):
    d_col_mesh_paths = {}
    links = kin_model.kinematics_parser._robot.link_map
    for k in links.keys():
        col = links[k].collisions
        for i in range(len(col)):
            m = col[i].geometry.mesh
            if m is not None:
                d_col_mesh_paths[k] = m.filename.replace("package://","")
    return d_col_mesh_paths

def get_fk_model_from_robot_yaml_file(robot_cfg_path = PATH_ROOT + '/data/curobo_cfg/franka_nofinger_tcp.yml'):
    import os

    from curobo.util_file import load_yaml 

    cfg_dict = load_yaml(robot_cfg_path)

    if not os.path.isabs(cfg_dict["robot_cfg"]["kinematics"]["urdf_path"]):
        cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = os.path.dirname(robot_cfg_path) + '/' + cfg_dict["robot_cfg"]["kinematics"]["urdf_path"]

        cfg_dict["robot_cfg"]["kinematics"]["asset_root_path"] = os.path.abspath(os.path.dirname(cfg_dict["robot_cfg"]["kinematics"]["urdf_path"]))

    robot_cfg = RobotConfig.from_dict(cfg_dict)

    kin_model = MyCudaRobotModel(robot_cfg.kinematics)

    update_get_col_mesh_paths_from_robot_model(kin_model)
    return kin_model


if __name__ == '__main__':
    import numpy as np

    from neurg.my_utils import np_tf
    from neurg.my_utils.util_time_measure import bench_torfunc_args_to_dic

    tensor_args = TensorDeviceType()
    # tensor_args = {'device': 'cuda', 'dtype': torch.float32}
    # fmt: off
    kin_model = get_fk_model()
    
    kin2 = get_fk_model_from_robot_yaml_file(robot_cfg_path = PATH_ROOT + '/data/curobo_cfg/franka_nofinger_tcp.yml')

    # kin_model.get_robot_link_meshes()
    # kin2.get_link_meshes()
    # fmt: off
    q_0 = np.array(
        [-0.0965195386873646, -0.48818972252726783, -0.5262399291814241, -2.0770811368204902,
         -0.23848617636484631, 1.6444697759066003, 0.23276192924707884])
    q_target = np.array(
        [0.09651953869727141, -0.48818972254287984, 0.5262399291691557, -2.0770811368275495, 
         0.23848617638248512, 1.6444697759146427, 1.3380343975414601])
    # fmt: on
    q = torch.tensor(q_0, dtype=torch.float32, device='cuda')

    qs = torch.rand(1000, 7, device='cuda')
    T_w2links0 = kin_model.fk_link_T44(qs)
    T_w2links1 = kin2.fk_link_T44(qs)
    diff = (T_w2links0 - T_w2links1).abs().max()
    print(f"diff = {diff}")

    T_w2links, T_w2ee = kin_model.fk_links_ee(q)

    link_T44 = kin_model.qp_fk_T44s(q)
    print(f"link_T44.shape = {link_T44.shape}")

    # check cal_J_ee_wrt_q
    J0 = cal_J_ee_wrt_q_raw(link_T44)
    J1 = cal_J_ee_wrt_q(link_T44)
    diff = (J0 - J1).abs().max()
    print(f"diff = {diff}")
    # ---------------------------------------------------

    print(np_tf.T2xyzrpy(link_T44[-2].cpu().numpy(), degrees=True).tolist())
    print(np_tf.T2xyzrpy(link_T44[-1].cpu().numpy(), degrees=True).tolist())
    print("-" * 100)
    ret = kin_model.qp_fk_T44s_J(q.squeeze(0))
    print(ret[0].shape)
    print(ret[1].shape)

    q = q.squeeze(0)

    with torch.no_grad():
        bench_torfunc_args_to_dic(kin_model.qp_fk_T44s_J, (q,), prefix="cu", repeat=10000, device='cuda')
    exit()
