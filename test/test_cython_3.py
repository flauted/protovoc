import unittest

from test.general_tests import NumericalizationTestSuite
from protovoc.numericalization.cython.cython_3 import Numericalization
from protovoc.vocab.cython import Vocab


class TestCython3Numericalization(unittest.TestCase, NumericalizationTestSuite):
    _voc = Vocab
    _num = Numericalization


if __name__ == "__main__":
    unittest.main()
