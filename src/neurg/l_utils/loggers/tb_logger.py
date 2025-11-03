import os
from typing import Any, Dict, Optional, Union

import pandas as pd
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch import loggers as pl_loggers
from torch import Tensor

from ..utils.util_log import get_time_str, logger_config


class MyTBLogger(pl_loggers.TensorBoardLogger):
    def __init__(
        self,
        save_dir="lightning_logs",
        name: Optional[str] = None,
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = False,
        prefix: str = "",
        sub_dir: Optional[_PATH] = None,
        time_start=None,
        **kwargs: Any,
    ):
        if isinstance(time_start, str):
            time_start_str = time_start
        else:
            time_start_str = get_time_str(time_start)

        ver_prefix = version or "v"
        version = "".join([ver_prefix, "_", time_start_str])
        super().__init__(
            save_dir,
            name,
            version,
            log_graph,
            default_hp_metric,
            prefix,
            sub_dir,
            **kwargs,
        )

        os.makedirs(self.log_dir, exist_ok=True)

        self.exp_logger = logger_config(
            os.path.join(self.root_dir, f"0_experiment.log"),
            __name__ + "_exp",
            stream=False,
        )

        self.exp_csv = os.path.join(self.root_dir, f"0_experiment.csv")

        txt_log_file = os.path.join(self.log_dir, f"{time_start_str}.log")
        self.txt_logger = logger_config(txt_log_file, __name__)

        self.log_info(
            "\n" + "-" * 80 + "\n"
            f"log to file: {txt_log_file}\n" + "—" * 47 + "\n" + f"│ {'Exp name':^20} │ {'Version':^20} │\n"
            f"│ {'—'*20} │ {'—'*20} │\n"
            f"│ {name:^20} │ {version:^20} │\n" + "—" * 47
        )
        # self.exp_name_v = f"{name}/{version}"
        self.exp_name_v = f"{version}"
        self.last_df = None

    def log_info(self, msg, on_exp=True, on_ver=True, stacklevel=2, plainout=False):
        find_logger = False
        if self.txt_logger is not None and on_ver:
            find_logger = True
            self.txt_logger.info(msg, stacklevel=stacklevel, plainout=plainout)
        if self.exp_logger is not None and on_exp:
            find_logger = True
            self.exp_logger.info(msg, stacklevel=stacklevel, plainout=plainout)
        if not find_logger:
            print(msg)

        self.last_df = None

    def is_same_columns(self, df: pd.DataFrame):
        if (
            self.last_df is not None
            and len(self.last_df.columns) == len(df.columns)
            and (self.last_df.columns == df.columns).all()
        ):
            return True
        else:
            return False

    def log_df(
        self,
        df_dict: Dict,
        on_exp=True,
        on_ver=True,
        stacklevel=2,
        rowshow=True,
        name=None,
        update=True,
        plainout=True,
        tablefmt="simple_outline",
    ):
        """
        rowshow : true:按行显示; false: 按列显示
        update: true: 根据exp_name_v更新； false: append df

        tablefmt: https://pypi.org/project/tabulate/
        """
        if isinstance(df_dict, dict):
            if not df_dict.get("exp_name_v", None):
                d1 = {"exp_name_v": self.exp_name_v}
                d1.update(df_dict)
                df_dict = d1

            dic = pd.json_normalize(df_dict, sep="/")
            dic = dic.iloc[0].to_dict()

            # dic = dic.astype(str)
            for k, val in dic.items():
                if isinstance(val, Tensor):
                    dic[k] = val.item() if val.numel() == 1 else val.tolist()

            df_item = pd.DataFrame(dic, index=[0])
        elif isinstance(df_dict, pd.DataFrame):
            df_item = df_dict
        else:
            print("Err log_df")

        update = update and "exp_name_v" in df_item

        if rowshow:
            msg_print = df_item.to_markdown(index=False, tablefmt=tablefmt)

            if plainout and self.is_same_columns(df_item):
                if "outline" in tablefmt:
                    msg_print = "\n".join(msg_print.split("\n")[3:])
                else:
                    msg_print = "\n".join(msg_print.split("\n")[2:])

            if not plainout:
                msg_print = "\n" + msg_print
        else:
            msg_print = df_item.set_index("exp_name_v").transpose().to_markdown(tablefmt=tablefmt)

        if self.txt_logger is not None and on_ver:
            find_logger = True
            self.txt_logger.info(msg_print, stacklevel=stacklevel, plainout=plainout)

        if self.exp_logger is not None and on_exp:
            find_logger = True
            self.exp_logger.info(msg_print, stacklevel=stacklevel, plainout=plainout)

        if on_exp:
            if name:
                fname = os.path.join(self.root_dir, f"{name}.csv")
            else:
                fname = self.exp_csv
            if os.path.exists(fname):
                # df = pd.read_csv(fname, index_col=0,sep='\\s*,', engine='python')

                # skip space
                df = pd.read_csv(fname, index_col=0)
                df.columns = df.columns.str.strip()
                for col in df.columns:
                    if pd.api.types.is_string_dtype(df[col]):
                        df[col] = df[col].str.strip()

                if update:
                    idx_raw = None
                    idxs = df[df["exp_name_v"] == df_item["exp_name_v"][0]].index
                    if len(idxs) == 1:
                        idx_raw = idxs[0]
                    elif len(idxs) > 1:
                        print("[warn]!! multilpe exp_name_v found!")

                    if idx_raw is not None:
                        d_raw = df.iloc[idx_raw].to_dict()
                        d_raw.update(df_item.iloc[0].to_dict())
                        df1 = pd.DataFrame(d_raw, index=[idx_raw])
                        df = pd.concat([df.drop(index=idx_raw), df1])
                else:
                    df = pd.concat([df, df_item], ignore_index=True)
            else:
                df = df_item

            df.to_csv(fname)
            df.to_excel(fname[:-3] + "xlsx")

        if not find_logger:
            print(msg_print)

        self.last_df = df_item


if __name__ == "__main__":
    import pandas as pd

    logger = MyTBLogger(name="debug")
    logger.log_info("This is normal out")
    logger.log_info("This is plainout.", plainout=True)
    logger.log_info("This is normal out")
    d0 = {"a": 5}
    logger.log_df(d0)

    d1 = {"a": 6, "info": "update=True"}
    logger.log_df(d1)

    d2 = {"a": 6, "info": "update=False"}
    logger.log_df(d2, update=False)

    d2 = {"a": 6, "b": 7, "c": 7}
    logger.log_df(d2, update=False)
    d2 = {"a": 6, "b": 8, "c": 8}
    logger.log_df(d2, update=False, plainout=True)
    d2 = {"a": 6, "b": 9, "c": 9}
    logger.log_df(d2, update=False, plainout=True)
    d2 = {"a": 6, "b": 9, "c": 9, "d": 9}
    logger.log_df(d2, update=False, plainout=True)
    d2 = {"a": 6, "b": 9, "c": 9, "d": 10}
    logger.log_df(d2, update=False, plainout=True)
    pass
