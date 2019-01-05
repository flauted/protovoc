import json

from benchmark import benchmarks
from protovoc.numericalization.numpy import Numericalization as NpNum
from protovoc.numericalization.numba import Numericalization as NbNum
from protovoc.numericalization.cython.cython_0 import Numericalization as Cy0Num
from protovoc.numericalization.cython.cython_1 import Numericalization as Cy1Num
from protovoc.numericalization.cython.cython_2 import Numericalization as Cy2Num
from protovoc.numericalization.cython.cython_3 import Numericalization as Cy3Num
from protovoc.vocab.basic import Vocab as BasicVoc
from protovoc.vocab.cython import Vocab as CyVoc


if __name__ == "__main__":
    benches = {
        "np": benchmarks(BasicVoc, NpNum, unk=False),
        "nb": benchmarks(BasicVoc, NbNum, unk=False),
        "cy0": benchmarks(BasicVoc, Cy0Num, unk=False),
        "cy1": benchmarks(CyVoc, Cy1Num, unk=False),
        "cy2": benchmarks(CyVoc, Cy2Num, unk=False),
        "cy3": benchmarks(CyVoc, Cy3Num, unk=False),
    }
    with open("benchmarks.json", "w") as f:
        json.dump(benches, f)
