from my_utils.util_apitable import dst_bulk_update_or_create, get_dst, dst2df
import numpy as np


def show_dst_metrics(dst_name: str):
    dst = get_dst(dst_name)
    df = dst2df(dst)

    n_exp = len(df["exp_ver"].unique())
    assert np.all(df.groupby("dist_method")["exp_ver"].count().to_numpy() == n_exp)

    print("=" * 10 + " " + dst_name + " " + "=" * 10)
    print(f"[{n_exp}] for all experiments " + "-" * 20)
    df1 = df.groupby("dist_method")[
        [
            "steps",
            "dis_close_traj_min",
        ]
    ].mean()
    df1["SR"] = df.groupby("dist_method")["success"].sum() / n_exp * 100
    if "dt_qp_all" in df.columns:
        df1["FPS"] = (
            df.groupby("dist_method")["dt_qp_all"]
            .mean()
            .apply(lambda x: 1 / x)
            .round(1)
        )
    else:
        df1["FPS"] = (
            df.groupby("dist_method")["dt.step_run"]
            .mean()
            .apply(lambda x: 1 / x)
            .round(1)
        )

    if 'rgbd_human' in dst_name:
        col_thresh=0.06
    elif "human" in dst_name:
        col_thresh = 0.1 - 0.02
    else:
        col_thresh = 0.05 - 0.02
    flag_df_reached = df["reached"] == 1
    flag_col = df["dis_close_traj_min"] <= col_thresh

    df["fail_by_qp"] = ~flag_df_reached
    df["fail_by_col"] = (flag_df_reached) & flag_col

    df1["fail_by_qp"] = df.groupby("dist_method")["fail_by_qp"].sum()
    df1["fail_by_col"] = df.groupby("dist_method")["fail_by_col"].sum()

    print(df1)
    # select all sucessful exp_ver
    flag_suc_exps = (
        df.groupby("exp_ver")["success"].sum() == 3
    )  # select exp_ver == flag_suc_exps
    df1 = df[df["exp_ver"].isin(flag_suc_exps[flag_suc_exps].index)]
    n_exp_full_suc = len(df1["exp_ver"].unique())
    print(f"[{n_exp_full_suc}] for fully-successed experiments " + "-" * 20)

    l_key = ["steps", "ee_move", "joint_move"]
    l_valid_key = [key for key in l_key if key in df1.columns]
    print(df1.groupby("dist_method")[l_valid_key].mean())


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(show_dst_metrics)
    # paper
    # show_dst_metrics('eval_qp_dyna_ball')
    # show_dst_metrics('eval_qp_shelf')
    # show_dst_metrics('eval_qp_cube')
    # show_dst_metrics('eval_qp_dyna_human')
    # show_dst_metrics('eval_mppi_rgbd_human')
    # show_dst_metrics('eval_mppi_dyna_ball')
    # show_dst_metrics('eval_mppi_rgbd_human_01310630')
