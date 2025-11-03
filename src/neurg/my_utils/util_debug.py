import os
import sys

def inDebug():
    if 'PYDEVD_LOAD_VALUES_ON_DEMAND' in os.environ:
        return True
    if sys.gettrace():
        return True
    return False