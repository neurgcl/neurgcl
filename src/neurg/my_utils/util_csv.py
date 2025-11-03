import os
import pandas as pd
import torch

from neurg.my_utils.util_log import get_time_str


class CSVLogger:
    def __init__(self, log_file, exp_name_v=None):
        self.log_file = log_file
        self.exp_name_v = exp_name_v or get_time_str()

    def __del__(self):
        pass

    def log_dict(self, df_dict, update=False, fname=None):
        """
        update: update columns according to exp_name
        """
        if isinstance(df_dict, dict):
            if not df_dict.get('index', None):
                d1 = {"exp_name_v": self.exp_name_v}
                d1.update(df_dict)
                df_dict = d1

            dic = pd.json_normalize(df_dict, sep='/')
            dic = dic.iloc[0].to_dict()

            # dic = dic.astype(str)
            for k, val in dic.items():
                if isinstance(val, torch.Tensor):
                    dic[k] = val.item() if val.numel() == 1 else val.tolist()

            df_item = pd.DataFrame(dic, index=[0])

            fname = self.log_file
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
                    idxs = df[df['exp_name_v'] == df_item['exp_name_v'][0]].index
                    if len(idxs) == 1:
                        idx_raw = idxs[0]
                    elif len(idxs) > 1:
                        print("[warn]!! multilpe exp_name_v found!")

                    if idx_raw is not None:
                        d_raw = df.iloc[idx_raw].to_dict()
                        d_raw.update(df_item.iloc[0].to_dict())
                        df1 = pd.DataFrame(d_raw, index=[idx_raw])
                        df = pd.concat([df.drop(index=idx_raw), df1])
                else:  # concat
                    df = pd.concat([df, df_item], ignore_index=True)
            else:
                df = df_item

            os.makedirs(os.path.dirname(fname), exist_ok=True)
            df.to_csv(fname)
            df.to_excel(fname[:-3]+"xlsx")
        else:
            print("not support")

        return df


if __name__ == "__main__":
    exp_name_v = get_time_str()
    csv_logger = CSVLogger(f'logs/test.csv', exp_name_v)
    d1 = {'batch_sz': 1,
          'dt_mean': 0.5}
    csv_logger.log_dict(d1)

    d1 = {'batch_sz': 1,
          'dt_mean': 0.5}
    csv_logger.log_dict(d1)
    d1 = {'batch_sz': 5,
          'dt_mean': 5.5}
    csv_logger.log_dict(d1)
