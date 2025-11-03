import os

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEV_WS_DIR = os.environ.get('DEV_WS_DIR',os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

if __name__ == "__main__":
    print(f"PKG_ROOT: {PKG_ROOT}")
    print(f"DEV_WS_DIR: {DEV_WS_DIR}")