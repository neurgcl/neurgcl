import itertools


def sweep_dict_to_list(sweep):
    param_combinations = list(itertools.product(*[sweep[key] for key in sweep]))
    keys = list(sweep.keys())
    l_param = []
    for val in param_combinations:
        param = {k: v for k, v in zip(keys, val)}
        l_param.append(param)
    return l_param
