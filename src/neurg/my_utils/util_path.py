import os
from pathlib import Path, PurePosixPath


class SPosixPath(Path, PurePosixPath):
    """Path subclass for non-Windows systems.

    On a POSIX system, instantiating a Path should return this object.
    """

    def __add__(self, other):
        if isinstance(other, str):
            return str(self) + other
        elif isinstance(other, Path):
            return str(self) + str(other)


class SPath(Path):
    def __new__(cls_raw, *args, **kwargs):
        cls = cls_raw
        if cls is Path or cls is SPath:
            cls = WindowsPath if os.name == 'nt' else SPosixPath
        self = cls._from_parts(args)
        if not self._flavour.is_supported:
            raise NotImplementedError("cannot instantiate %r on your system" % (cls.__name__,))

        # self.__add__ = partial(cls_raw.__add__, self)
        return self

    # def __add__(self, other):
    #     if isinstance(other, str):
    #         return SPath(str(self) + other)
    #     elif isinstance(other, Path):
    #         return SPath(str(self) + str(other))


if __name__ == "__main__":
    PATH_ROOT = SPath(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../')))
    rel_path = os.path.relpath('data/meshs')
    rel = Path('a')
    print(f"PATH_ROOT: {PATH_ROOT}")
    print(f"PATH_ROOT: {PATH_ROOT+'a'}")
    print(f"PATH_ROOT: {PATH_ROOT + rel_path}")
    print(f"PATH_ROOT: {PATH_ROOT + rel}")
