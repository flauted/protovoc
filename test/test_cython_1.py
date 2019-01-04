import unittest

from test.general_tests import NumericalizationTestSuite
from protovoc.numericalization.cython.cython_1 import Numericalization
from protovoc.vocab.cython import Vocab


class TestCython1Numericalization(unittest.TestCase, NumericalizationTestSuite):
    _voc = Vocab
    _num = Numericalization


if __name__ == "__main__":
    unittest.main()
