import os
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_file import load_json, write_json


@dataclass_json
@dataclass
class DatasetCfg:
    version: str
    mesh_dir: str
    link_meshs: list[str]
    N_REPEAT_SAMPLE: int

    def write_json(self, path: str):
        cfg = self.to_dict()
        if os.path.isabs(self.mesh_dir):
            mesh_dir = os.path.relpath(self.mesh_dir, PATH_ROOT)
            cfg["mesh_dir"] = str(mesh_dir)
        write_json(path, cfg)

    @classmethod
    def read_json(cls, path: str) -> "DatasetCfg":
        cfg = load_json(path)
        if not os.path.isabs(cfg["mesh_dir"]):
            cfg["mesh_dir"] = os.path.abspath(
                os.path.join(PATH_ROOT, cfg["mesh_dir"])
            )
        return cls.from_dict(cfg)

    def get_link_names(self):
        return [fname.split(".")[0] for fname in self.link_meshs]


if __name__ == "__main__":
    config = DatasetCfg(
        version="asda",
        mesh_dir="logs/nsdf/dataset/meshes",
        link_meshs=[
            "hand.stl",
            "link0.stl",
            "link1.stl",
            "link2.stl",
            "link3.stl",
            "link4.stl",
            "link5.stl",
            "link6.stl",
            "link7.stl",
        ],
        N_REPEAT_SAMPLE=500,
    )

    config.write_json(PATH_ROOT / "logs/tmp/test_config.json")
    ret = DatasetCfg.read_json(PATH_ROOT / "logs/tmp/test_config.json")
    print(ret)
    print(ret.get_link_names())
