from copy import deepcopy

import numpy as np
import pybullet as op
from bl_sim_env.env.bl_extras_base import BLPanda
from qp_ik.env.col_env_base import ColEnvBase


class BallEnv(ColEnvBase):
    def __init__(self, robot: BLPanda, name:str="ball", ball_r:float=0.05):
        ColEnvBase.__init__(self, robot, name)
        self.ball_r = ball_r

    def init_obstacles(self, l_pos):
        """
        l_pos: n, 3
        """
        p = self.p
        l_pos = np.array(l_pos)

        if self.bl_col_shape_id is not None:
            pass
        else:
            self.bl_col_shape_id = p.createCollisionShape(
                op.GEOM_SPHERE, radius=self.ball_r
            )
        if len(l_pos.shape) == 2:
            self.bl_objs = [
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=self.bl_col_shape_id,
                    basePosition=_pos,
                )
                for _pos in l_pos
            ]
        self.pcs = l_pos

    def get_pcs_with_offset(self, offset, **kwargs):
        """
        offset: n, 3
        return: n_pt, 3
        """
        return deepcopy(offset)
    def get_bl_dist_min(self):
        dists = super().gel_bl_dists()
        dists = dists[dists != 0]
        if len(dists) == 0:
            return np.nan
        else:
            return dists.min()

if __name__ == "__main__":
    from bl_sim_env.env.panda_env_base import BLPanda, PandaEnv

    col_type = "ball"

    env = PandaEnv(col_obj_name=col_type)
    p = env.p
    robot: BLPanda = env.robot
    col_env = BallEnv(robot, 0.05)
    env.col_env = col_env

    q_0 = np.array(
        [
            -0.0965195386873646,
            -0.48818972252726783,
            -0.5262399291814241,
            -2.0770811368204902,
            -0.23848617636484631,
            1.6444697759066003,
            0.23276192924707884,
        ]
    )
    robot.reset_q(q_0)

    col_env.update_obstacles([[0.3, 0, 0.3], [0.3, 0.2, 1], [10, 10, 10]])

    bl_dist = col_env.gel_bl_dists()  # [n_obj, n_link(9)]
    print(bl_dist)
