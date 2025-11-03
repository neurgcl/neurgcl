from abc import abstractmethod
from copy import deepcopy

import numpy as np
from bl_sim_env.env.bl_extras_base import BLPanda


class ColEnvBase:
    def __init__(self, robot: BLPanda, name:str="col_env_base"):
        self.robot = robot
        self.p = robot.p
        self.name = name

        self.bl_col_shape_id = None
        self.bl_objs = []
        self.ball_r = 0.0

        self._pcs_raw = None  # raw pts of mesh (n_pt, 3)
        self.pcs = None  # current transformed pcs (n_pt, 3)

    @abstractmethod
    def init_obstacles(self, l_pos):
        """
        create obstacles

        Args:
            l_pos: n_obj, 3  (obstacle centers)
        """
        raise NotImplementedError

    def delete_obstacles(self):
        for obj in self.bl_objs:
            self.p.removeBody(obj)
        self.bl_objs = []


    def _update_pcs(self, l_pos):
        """
        l_pos: n_obj, 3
        """
        self.pcs = self.get_pcs_with_offset(l_pos)

    def update_obstacles(self, l_pos, **kwargs):
        """
        l_pos: n, 3
        """
        p = self.p
        if l_pos is not None and len(self.bl_objs) == len(l_pos):
            for i, obj in enumerate(self.bl_objs):
                p.resetBasePositionAndOrientation(obj, l_pos[i], [0, 0, 0, 1])
            l_pos = np.array(l_pos)
            self._update_pcs(l_pos)
        else:
            self.delete_obstacles()

            l_pos = np.array(l_pos)
            self.init_obstacles(l_pos)
            self._update_pcs(l_pos)

    @abstractmethod
    def get_pcs_with_offset(self, offset, **kwargs):
        """
        offset: n_obj, 3
        return: n_pt, 3
        """
        raise NotImplementedError

    def get_pcs(self):
        """
        return: n_pt, 3
        """
        return deepcopy(self.pcs)

    def gel_bl_dists(self):
        """
        Return:
            dists: [n_obj, n_xx]  todo:check
        """
        dists = []
        for obj_id in self.bl_objs:
            closestPoints = self.p.getClosestPoints(
                self.robot.id_robot,
                obj_id,
                10,
            )
            if len(closestPoints) == 0:
                dists.append([np.nan])
            else:
                #  # [3] link id A, [4]link id B, [8] distance
                # closestDis = np.array([(x[3], x[4], x[8]) for x in closestPoints])
                dis_to_links = np.array([x[8] for x in closestPoints])
                dists.append(dis_to_links)
        dists = np.array(dists)
        return dists
