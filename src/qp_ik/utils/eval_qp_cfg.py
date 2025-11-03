import enum
import os
import re
from dataclasses import asdict, dataclass, field
from typing import ClassVar, Final, List, Union

from bl_sim_env.utils.util_file import load_pickle
from bl_sim_env.utils.util_log import get_time_str
from qp_ik.cfg import DEV_WS_DIR


def parse_version_key_from_info_path(info_path):
    version_key = None
    folder, filename = os.path.split(info_path)
    # '250117_234140_jsdf_traj_info.pkl' - > '250117_234140_jsdf
    try:
        version_key = re.match("(.*)_traj_info\.pkl", filename).groups()[0]
    except:
        pass
    return version_key


def parse_exp_ver_from_info_path(info_path):
    version_key = parse_version_key_from_info_path(info_path)
    return version_key[: len("250117_234140")]

def parse_method_from_info_path(info_path):
    version_key = parse_version_key_from_info_path(info_path)
    return version_key[len("250117_234140_"):]

# test_ver/exp_ver
@dataclass
class EvalQPArgs:
    col_type: str = "col_type"
    controller:str = "qp"
    dst_name: str = ""
    test_ver: str = ""
    exp_ver: str = ""
    csv_file: str = ""
    save_video: bool = True
    log_dir: str = ""
    save_config: Union[bool, str] = False
    max_step: int = 300
    use_gui: bool = True
    flag_show_txt:bool = True

    methods: List[str] = field(default_factory=lambda: ["nsdf"])
    recover_info_path: str = ""

    full_test_name = ""
    env_init_args = None

    def __post_init__(args):
        args.exp_ver = args.exp_ver or get_time_str()
        test_ver = args.test_ver
        if test_ver and not test_ver.startswith("_"):
            test_ver = "_" + args.test_ver

        args.full_test_name = f"{args.col_type}{test_ver}"
        if not args.csv_file and args.test_ver:
            args.csv_file = DEV_WS_DIR + f"/logs/record/{args.controller}_{args.full_test_name}.csv"
        if not args.log_dir:
            if args.controller == 'mppi':
                args.log_dir = DEV_WS_DIR + f"/logs/{args.controller}/{args.full_test_name}"
            else:
                args.log_dir = DEV_WS_DIR + f"/logs/qpik/{args.full_test_name}"
        
        if not args.dst_name:
            args.dst_name = (
                f"eval_{args.controller}_{args.col_type}"
                if args.test_ver
                else f"eval_{args.controller}"
            )
            
        args.recover_info()

    def recover_info(args):
        if args.recover_info_path:
            info_path = args.recover_info_path
            d_info = load_pickle(info_path)
            env_init_args = {
                "q_0": d_info["q_0"],
                "T_target": d_info["T_target"],
                "pts_traj": d_info["pts_traj"],
            }
            if "col_env_args" in d_info:
                env_init_args["col_env_args"] = d_info["col_env_args"]
            args.env_init_args = env_init_args


def parse_my_args(default_args: dict = None, args=None) -> EvalQPArgs:
    import sys

    import yaml
    from jsonargparse import ArgumentParser, dict_to_namespace

    parser = ArgumentParser()
    parser.add_class_arguments(EvalQPArgs)

    if args is None:
        args = sys.argv[1:]
    else:
        args = list(args) + sys.argv[1:]

    ns = None
    if default_args is not None:
        ns = dict_to_namespace(default_args)
    args = parser.parse_args(args, namespace=ns, defaults=False)

    args = EvalQPArgs(**args)  # run __post_init__

    if args.save_config and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        with open(args.log_dir + "/config.yaml", "w") as f:
            d_save = asdict(args)
            f.write(yaml.dump(d_save))
    return args


if __name__ == "__main__":
    args = parse_my_args({"col_type": "ball"})
    print(args)
