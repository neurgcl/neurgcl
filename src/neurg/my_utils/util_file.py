import json
import os
import re

import _pickle as pickle
import yaml


def detect_walk_files(dir_path, suffix=None, re_pattern=None):
    """_summary_

    Args:
        dir_path (_type_): _description_
        suffix (_type_, optional): _description_. Defaults to None.
        re_pattern (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: list of file paths
    """
    l_files = []

    # Get all files
    for root, dirs, files in os.walk(dir_path):
        fnames = [os.path.join(root, fname) for fname in files]
        l_files.extend(fnames)

    # Filter with suffix
    if suffix is not None:
        # re_suffix = ".*"+re.escape(suffix)+"$"
        re_suffix = ".*" + suffix + "$"
        l_files = list(filter(lambda fname: re.match(re_suffix, fname), l_files))

    if re_pattern is not None:
        l_files = list(filter(lambda fname: re.match(re_pattern, fname), l_files))

    l_files.sort()
    return l_files


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_json(path, dic, indent=4):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(dic, f, indent=indent)


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    return None


def load_yaml(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data
    return None


def write_yaml(path, dic, indent=4):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(dic, f, default_flow_style=False, indent=indent)
