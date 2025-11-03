from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_sweep import sweep_dict_to_list
from neurg.my_utils.util_file import detect_walk_files


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--cpus', type=int, default=1)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--log_name', type=str, default=None)
    args = parser.parse_args()
    data_dir = args.data_dir
    out_file = args.out_file or (PATH_ROOT + '/logs/cmd_train_all.sh')
    log_name = args.log_name or "nsdf"

    data_files = detect_walk_files(data_dir+'/train', '.pkl')

    link_names = []
    for df in data_files:
        link_name = df.split('/')[-1].split('.')[0]
        link_names.append(link_name)

    sweep = {
        'actv_type': ['softplus'],
        'hid_dim': [32],
        'hid_size': [3],
        'link_name': link_names,
    }
    l_param = sweep_dict_to_list(sweep)

    l_cmds = []
    for param in l_param:
        actv_type = param['actv_type']
        hid_dim = param['hid_dim']
        hid_size = param['hid_size']
        link_name = param['link_name']

        comm_cmd = f"src/neurg/net/l_model_linksdf.py --config=config/lsdf_fast_train.yaml --model.hid_dim={hid_dim} --model.hid_size={hid_size} --model.actv_type={actv_type}"

        surfix = f"{log_name}/{actv_type}/coll/{hid_dim}x{hid_size}/"
        cmd = f'python3 {comm_cmd} --logger.name={surfix}{link_name} --data.init_args.link_name={link_name}'
        l_cmds.append(cmd)

    with open(out_file, 'w') as f:
        f.write('\n'.join(l_cmds) + '\n')
