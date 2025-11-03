import matplotlib.pyplot as plt
import numpy as np

from bl_sim_env.env.panda_env_base import BLPanda, PandaEnv
from neurg.my_utils import np_tf
from qp_ik.ca_qp_ik_cpp import CAQPIK
from qp_ik.env.ball_env import BallEnv
from qp_ik.utils import robot_kin
from qp_ik.utils.eval_qp import run_log_one_episode
from qp_ik.utils.eval_qp_cfg import EvalQPArgs, parse_my_args
from qp_ik.utils.traj import gen_line_traj_constant_v
from qp_ik.utils.util_log_info import print_init_target_info

np.set_printoptions(precision=6, suppress=True)

def run_single_eval(args:EvalQPArgs):
    col_type = args.col_type

    # Env --------------------
    env = PandaEnv()
    env.flag_show_txt = args.flag_show_txt
    env.reset_camera(distance=1.5, yaw=90, pitch=-0, cameraTargetPosition=[0, 0, 0.5])
    env.pos_debug_txt = [0.7, 0, 0]

    robot: BLPanda = env.robot
    col_env = BallEnv(robot, col_type, 0.05)
    env.col_env = col_env

    # IK --------------------
    ik = CAQPIK(col_type=col_type)
    ik.ignore_vel_limit = False
    ik.check_col = True
    ik.col_margin = 0.05
    ik.col_r = 0.05
    ik.q_vel_limit = np.array([2.1750, 2.1750, 2.1750,  2.1750, 2.6100, 2.6100, 2.6100])*ik.vel_scale # ros panda
    ik.q_step_limit = np.deg2rad(2)
    ik.filter_Gamma = False
    ik.init_QP()
    ik.qp_options = {}
    ik.qp_options = {
        "printLevel": "none",
        # "jit":True,
        # "sparse": True
    }
    # fmt: off

    if not args.env_init_args:
        q_0 = np.array(
            [-1, -0.48818972252726783, -0.5262399291814241, -2.0770811368204902,
            -0.23848617636484631, 1.6444697759066003, 0.23276192924707884])
        q_1 = np.array(
            [0.8, -0.48818972254287984, 0.5262399291691557, -2.0770811368275495, 
            0.23848617638248512, 1.6444697759146427, 1.3380343975414601])
        # fmt: on

        T_0 = robot_kin.fk_tfs(q_0)[-1]
        T_target = robot_kin.fk_tfs(q_1)[-1]

        # random init --------------------
        q_0 = robot.get_ik(T_0)

        # random target --------------------
        q_0 = np.array([-0.5, 0.017361, -0.7, -1.865001, -0.012008, 1.87805, 1.579455])
        robot.reset_q(q_0)
        T_target = np_tf.xyzrpy2T([0.4, 0.5, 0.4, 180, 0, 0], degrees=True)

        pts = np.array([[0.3, -0.2, 0.6], [0.3, 0.2, 0.4]])
        col_env.update_obstacles(pts)

        # fmt: off
        pts[:, 2] += (np.random.random(2)*2-1) * 0.03 # random z
        pts[:, 1] += (np.random.random(2)*2-1) * 0.03 # random y
        pts_traj = pts.copy()[np.newaxis, :].repeat(args.max_step, axis=0)  # (n_step, n_sphere, 3)
        random0 = np.random.random()
        random1 = np.random.random()
        pts_traj[:,0,2] += gen_line_traj_constant_v(random0+0.0, 0.3, 0.15, 0.01, args.max_step)
        pts_traj[:,1,2] += gen_line_traj_constant_v(random1+0.5, 0.3, 0.15, 0.01, args.max_step)
        # fmt: on
    else:
        q_0 = args.env_init_args["q_0"]
        T_target = args.env_init_args["T_target"]
        pts_traj = args.env_init_args["pts_traj"]

    # print info --------------------
    print_init_target_info(robot, robot_kin, q_0, T_target)


    robot.reset_q(q_0)
    col_env.update_obstacles(pts_traj[0])

    for method in args.methods:
        ik.change_method(method)
        ret = run_log_one_episode(
            env,
            ik,
            q_0,
            T_target,
            pts_traj,
            max_step=args.max_step,
            save_video=args.save_video,
            exp_ver_prefix=args.exp_ver,
            log_dir=args.log_dir,
            quiet=True,
            args=args,
        )

    if not args.save_video:
        plt.show()

    del env
    print("done")

if __name__ == "__main__":
    default_args = dict(
        col_type="dyna_ball",
        max_step=1000,
        methods=["nsdf", "jsdf", "rdf"],
    )

    args = parse_my_args(default_args)
    print(args)
    run_single_eval(args)
