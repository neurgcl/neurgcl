import os

PATH_ROOT = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../..'))

DIR_DATA = os.path.abspath(os.path.join(PATH_ROOT, 'data'))

if __name__ == "__main__":
    print(f"PATH_ROOT: {PATH_ROOT}")
