from my_utils.util_apitable import dst_bulk_update_or_create, get_dst
import pandas as pd
from qp_ik.cfg import DEV_WS_DIR

def upload_csv_to_apitable(
        csv_file:str=DEV_WS_DIR + '/' + 'logs/record/dyna_human_01210509.csv',
        dst_name:str='eval_qp',
):
    df = pd.read_csv(csv_file, index_col=0)

    df['test_ver'] = df['test_ver'].astype(str)
    df['test_ver'] = df['test_ver'].apply(lambda x: '0' + x if len(x) == 7 else x)

    df.set_index('exp_ver_w_method', inplace=True)
    dst = get_dst(dst_name)
    dst_bulk_update_or_create(dst, df)


if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(upload_csv_to_apitable)
    # upload_csv_to_apitable(DEV_WS_DIR + '/' + 'logs/record/dyna_human_01210509.csv','eval_qp_dyna_human')
