import os
from pathlib import Path

from neurg.my_utils.util_path import SPath

PATH_ROOT = SPath(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../..')))
DIR_DATA = SPath(os.path.abspath(os.path.join(PATH_ROOT, 'data')))


if __name__ == "__main__":
    print(f"PATH_ROOT: {PATH_ROOT}")
    print(f"DIR_DATA: {DIR_DATA}")
