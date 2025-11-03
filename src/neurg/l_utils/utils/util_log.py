import datetime
import json
import logging
import re
import sys
from collections import OrderedDict
from typing import Dict

import numpy as np
import pandas as pd

DEF_TIME_FMT_ = '%y%m%d_%H%M%S'


def listd_2_d(l):
    if len(l) == 0:
        return {}
    keys = l[0].keys()
    d = {}
    for k in keys:
        d[k] = []
        for item in l:
            d[k].append(item[k])

    for k in keys:
        d[k] = np.array(d[k])
    return d


def get_time_str(time=None):
    if time is None:
        time = datetime.datetime.now()
    return time.strftime(DEF_TIME_FMT_)


def get_time_str_mdt(time=None):
    """mdt: %m%d_%H%M%S"""
    if time is None:
        time = datetime.datetime.now()
    return time.strftime('%m%d_%H%M%S')


def match_stream_handler(handler, streams=[]):
    """
    Identify stream handlers writing to the given streams(s).

    :param handler: The :class:`~logging.Handler` class to check.
    :param streams: A sequence of streams to match (defaults to matching
                    :data:`~sys.stdout` and :data:`~sys.stderr`).
    :returns: :data:`True` if the handler is a :class:`~logging.StreamHandler`
              logging to the given stream(s), :data:`False` otherwise.

    This function can be used as a callback for :func:`find_handler()`.
    """
    return isinstance(handler, logging.StreamHandler) and getattr(handler, 'stream') in (
        streams or (sys.stdout, sys.stderr)
    )


def match_file_handler(handler):
    return isinstance(handler, logging.FileHandler)


def logger_df_longline(logger, df, df_name=None):
    pd_display_width = pd.get_option('display.width')
    # pd.set_option('display.width', 5000)
    str_out = ''
    if df_name:
        str_out += df_name
    str_out += '\n' + str(df)
    logger.info(str_out)
    pd.set_option('display.width', pd_display_width)


def clear_irc_color(string):
    pattern = r'\x1b(\[.*?[@-~]|\].*?(\x07|\x1b\\))'
    return re.sub(pattern, '', string)


def sort_dict(d):
    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def format_dict(d: Dict, stdout: bool = True, sort=False, fmt_json=True):
    ret = ""
    if sort:
        d = sort_dict(d)

    if fmt_json:
        d1 = {}
        cur_d = d1
        max_len = 0
        for k, v in d.items():
            try:
                v = float(v)
            except:
                pass
            keys = k.split('/')
            cur_d = d1
            for i, new_key in enumerate(keys):
                if i == len(keys) - 1:
                    cur_d[new_key] = v
                    max_len = max(max_len, len(cur_d.keys()))
                else:
                    if not cur_d.get(new_key, None):
                        cur_d[new_key] = {}
                    cur_d = cur_d[new_key]

        if max_len <= 1:
            for k, v in d.items():
                ret += f"{k} = {v}\n"
        else:
            ret += json.dumps(d1, indent=4)
            ret += "\n"
    else:
        for k, v in d.items():
            ret += f"{k} = {v}\n"

    if len(ret) >= 1 and ret[-1] == '\n':
        ret = ret[:-1]

    if stdout:
        print(ret)
    return ret


class PlainLogger:
    def __init__(self, logger) -> None:
        self.logger: logging.Logger = logger
        self.fmts_old = [h.formatter for h in logger.handlers]
        self.fmt_plain = logging.Formatter('%(message)s')

    def info(self, msg, *args, **kwargs):
        stacklevel = kwargs.get('stacklevel', 1)
        stacklevel += 1
        kwargs.update({'stacklevel': stacklevel})

        plainout = kwargs.get('plainout', None)
        if plainout is not None:
            kwargs.pop('plainout')

        stream = kwargs.get('stream', None)
        if stream is not None:
            kwargs.pop('stream')

        if plainout:
            for h in self.logger.handlers:
                h.setFormatter(self.fmt_plain)

            self.logger.info(msg, *args, **kwargs)

            for h, fmt in zip(self.logger.handlers, self.fmts_old):
                h.setFormatter(fmt)
        else:
            self.logger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        stacklevel = kwargs.get('stacklevel', 1)
        stacklevel += 1
        kwargs.update({'stacklevel': stacklevel})
        self.logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        stacklevel = kwargs.get('stacklevel', 1)
        stacklevel += 1
        kwargs.update({'stacklevel': stacklevel})
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        stacklevel = kwargs.get('stacklevel', 1)
        stacklevel += 1
        kwargs.update({'stacklevel': stacklevel})
        self.logger.warning(msg, *args, **kwargs)


def logger_config(
    filename=None,
    name=None,
    stream=True,
) -> logging.Logger:
    import coloredlogs
    from coloredlogs import find_handler, replace_handler

    '''
    配置log
    :param filename: 输出log路径
    :param name: 记录中name，可随意
    :return:
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)

    # 删除重复的handler
    replace_handler(logger, match_file_handler, True)
    replace_handler(logger, match_stream_handler, True)

    # setup FileHandler
    if filename is not None:
        file_h = logging.FileHandler(filename, encoding='UTF-8')
        file_h.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(filename)s[l:%(lineno)-4d] - %(levelname)s: %(message)s')
        file_h.setFormatter(file_formatter)
        logger.addHandler(file_h)

    # isinstance(file_h, logging.StreamHandler)
    # type(file_h) == logging.StreamHandler

    # match_stream_handler(file_h)
    # isinstance(file_h, logging.StreamHandler)
    # getattr(file_h, 'stream') in (sys.stdout, sys.stderr)

    if stream:
        # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
        # stream_h = logging.StreamHandler()
        # stream_h.setLevel(logging.INFO)
        # stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        # stream_h.setFormatter(stream_formatter)
        # logger.addHandler(stream_h)

        coloredlogs.install(
            logger=logger,
            fmt='%(asctime)s - %(filename)s:%(lineno)-4d - %(levelname)s: %(message)s',
            level=logging.INFO,
        )

    # handler, other_logger = find_handler(logger, match_stream_handler)

    return PlainLogger(logger)


if __name__ == "__main__":
    logger = logger_config(filename='log.log', name='')
    logger.info("info")
    logger.error("error")
    logger.debug("debug")
    logger.warning("warning")
    print('print和logger输出是有差别的！')

    fmters_old = [h.formatter for h in logger.handlers]
    fmt_new = logging.Formatter('%(message)s')

    for h in logger.handlers:
        h.setFormatter(fmt_new)

    logger.info("info")

    for h, fmt in zip(logger.handlers, fmters_old):
        h.setFormatter(fmt)

    logger = logger_config(filename='log.log', name='')
    logger.info("info")
    logger.error("error")
    logger.debug("debug")
    logger.warning("warning")
