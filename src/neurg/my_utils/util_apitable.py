import os
from typing import List

import numpy as np
import pandas as pd
from apitable import Apitable
from apitable.const import MAX_GET_RECORDS_PRE_REQ, MAX_WRITE_RECORDS_PRE_REQ
from apitable.datasheet import Datasheet
from tqdm import tqdm

from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_file import load_yaml

_config = load_yaml(PATH_ROOT + '/config/apit.yaml')
if _config is not None:
    APITABLE_TOKEN = _config['APITABLE_TOKEN']
    APITABLE_API_BASE = _config['APITABLE_API_BASE']
    DEF_SPACE_ID = _config['APITABLE_DEF_SPACE_ID']
else:
    APITABLE_TOKEN = os.environ.get('APITABLE_TOKEN')
    APITABLE_API_BASE = os.environ.get('APITABLE_API_BASE')
    DEF_SPACE_ID = os.environ.get('APITABLE_DEF_SPACE_ID')


apitable = Apitable(APITABLE_TOKEN, api_base=APITABLE_API_BASE)


def is_number(s):
    if isinstance(s, (int, float)):
        return True
    if isinstance(s, str):
        return s.isdecimal()
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def split_list(l, chunk_size=MAX_WRITE_RECORDS_PRE_REQ):
    return [l[i : i + chunk_size] for i in range(0, len(l), chunk_size)]


def is_dict_contained(small_dict, large_dict):
    return all(large_dict.get(key) == value for key, value in small_dict.items())


def dst_bulk_update_or_create(dst: Datasheet, _df: pd.DataFrame, verbose=False):
    # add new columns
    fields_name = [x.name for x in dst.fields.all()]
    _df_flat = None
    # convert bool to int
    idx_cols = _df.columns[_df.dtypes == bool]
    if len(idx_cols) > 0:
        _df = _df.copy()
        _df[idx_cols] = _df[idx_cols].astype(int)

    _df_idx_is_none = _df.index.names == [None]
    _df_idx_names = list(_df.index.names)
    if _df_idx_is_none:
        _df_idx_names.remove(None)
        _df_flat = _df
        df_col_names = _df.columns.tolist()
    else:
        _df_flat = _df.reset_index()
        df_col_names = _df_idx_names + _df.columns.tolist()

    for col in df_col_names:
        if not col in fields_name:
            _drop_na = _df_flat[col].dropna()
            if len(_drop_na) > 0:
                if is_number(_drop_na.iloc[0]):
                    precision = 2
                    try:
                        dtype = pd.to_numeric(_drop_na).dtype
                        if np.issubdtype(dtype, np.integer):
                            precision = 0
                    except Exception as e:
                        print("Exception", e)
                        continue

                    print(f"append number field: {col}")
                    dst.fields.create(get_num_field(col, precision=precision))
                elif isinstance(_drop_na.iloc[0], bool):
                    print(f"append boolean field: {col}")
                    dst.fields.create(get_num_field(int(col), precision=0))
                else:
                    print(f"append text field: {col}")
                    dst.fields.create(get_singletext_field(col))

    # get records in dst
    _df_in_dst = dst2df(dst, rec_id=True)
    idx_names = list(_df.index.names)
    try:
        _df_in_dst.set_index(idx_names, inplace=True)
    except:
        pass

    # flag_idx_in_dst = _df.index.isin(_df_in_dst.index)
    # idx_in_dst = _df.index[flag_idx_in_dst]
    # idx_not_in_dst = _df.index[~flag_idx_in_dst]

    l_update_recs = []
    l_create_recs = []

    for i in range(len(_df)):
        item = _df.iloc[i].dropna()
        idx = item.name
        dic = item.to_dict()
        if _df_idx_is_none:
            l_create_recs.append(dic)
        else:
            if idx in _df_in_dst.index:
                fields = dic
                recordId = _df_in_dst.loc[idx]['recordId']
                dic_in_dst = _df_in_dst.loc[idx].to_dict()

                # if not set(dic.items()).issubset(dic_in_dst.items()):
                if not is_dict_contained(dic, dic_in_dst):
                    l_update_recs.append({'recordId': recordId, 'fields': fields})
            else:
                # l_create_recs.append(_df.iloc[[i]].dropna().reset_index().to_dict('records')[0])
                # l_create_recs.append(_df.iloc[[i]].dropna(axis=1).reset_index().to_dict('records')[0])
                l_create_recs.append(_df.iloc[[i]].replace({np.nan: None}).reset_index().to_dict('records')[0])

    for l in tqdm(split_list(l_update_recs), desc='update', disable=not verbose):
        dst.records.bulk_update(l)

    for l in tqdm(split_list(l_create_recs), desc='create', disable=not verbose):
        dst.records.bulk_create(l)

    if verbose:
        print(f"update records: {len(l_update_recs)}")
        print(f"create records: {len(l_create_recs)}")


def dst_id2dst(dst_id, space_id=None):
    space_id = space_id or DEF_SPACE_ID
    return apitable.space(space_id).datasheet(dst_id)


def get_dst_id(name, space_id=None):
    space_id = space_id or DEF_SPACE_ID
    for node in apitable.space(space_id).nodes.search(type='Datasheet'):
        if node.name == name:
            dst_id = node.id
            return dst_id
    return None


def get_dst(name, space_id=None):
    space_id = space_id or DEF_SPACE_ID
    dst_id = get_dst_id(name, space_id)
    if dst_id is None:
        return None
    return apitable.space(space_id).datasheet(dst_id)


def dst_append_dfcol(dst, df):
    fields_name = [x.name for x in dst.fields.all()]
    columns = df.columns.tolist()
    for col in columns:
        if not col in fields_name:
            if len(df[col].dropna()) > 0:
                if is_number(df[col].dropna().iloc[0]):
                    print(f"append number field: {col}")
                    dst.fields.create(get_num_field(col, precision=2))
                else:
                    print(f"append text field: {col}")
                    dst.fields.create(get_singletext_field(col))


def dst2df(dst_id: str, rec_id=False) -> pd.DataFrame:
    if isinstance(dst_id, str):
        dst = dst_id2dst(dst_id)
    elif isinstance(dst_id, Datasheet):
        dst = dst_id
    else:
        raise TypeError(f"dst_id type error: {type(dst_id)}")
    fields = dst.get_fields()
    columns = [x['name'] for x in fields.model_dump()['data']['items'] if not (x['type'] in ['Attachment', 'URL'])]
    records = dst.get_records_all()
    datas = [rec.data for rec in records]
    if rec_id:
        recordIds = [rec.id for rec in records]
        for i, id in enumerate(recordIds):
            datas[i]['recordId'] = id
        columns = ['recordId'] + columns

    df = pd.DataFrame(datas)
    df_col = df.columns.tolist()

    # remove columns not in df_col
    col_new = [col for col in columns if col in df_col]
    df = df[col_new].copy()
    return df


def dstname2df(name):
    dst = get_dst(name)
    df = dst2df(dst)
    return df


def get_singletext_field(name, default=''):
    prop = {
        'defaultValue': default,
    }

    req_data = {
        'name': name,
        'type': 'SingleText',
        'property': prop,
    }
    return req_data


def get_singleselecr_field(name, default=''):
    prop = {
        'defaultValue': default,
    }

    req_data = {
        'name': name,
        'type': 'SingleSelect',
        'property': prop,
    }
    return req_data


def get_num_field(name, default='', precision=0):
    prop = {'defaultValue': default, 'precision': precision}

    req_data = {'name': name, 'type': 'Number', 'property': prop}
    return req_data


def update_dst_by_index(dst: Datasheet, df: pd.DataFrame):
    idx_names = list(df.index.names)
    for i in range(len(df)):
        item = df.iloc[i].dropna()
        idx = item.name
        # dic= {k: v for k, v in zip(idx_names, idx)}
        dic = item.to_dict()
        if len(idx_names) > 1:
            keys = {k: v for k, v in zip(idx_names, idx)}
        else:
            keys = {idx_names[0]: idx}
        dst.records.update_or_create(dic, **keys)


def update_dst_by_keys(dst: Datasheet, df: pd.DataFrame, keynames: List[str]):
    for i in range(len(df)):
        item = df.iloc[i].dropna()
        dic = item.to_dict()
        keys = {key: dic[key] for key in keynames}
        dst.records.update_or_create(dic, **keys)


if __name__ == '__main__':
    dst = get_dst('test')
    df = pd.DataFrame([{'repeat': 1, 'warmup': 4}])
    df.set_index('a', inplace=True)
    dst_bulk_update_or_create(dst, df, verbose=True)
