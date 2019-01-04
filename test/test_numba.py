import unittest

from test.general_tests import NumericalizationTestSuite
from protovoc.numericalization.numba import Numericalization
from protovoc.vocab.basic import Vocab


class TestNumbaNumericalization(unittest.TestCase, NumericalizationTestSuite):
    _voc = Vocab
    _num = Numericalization


if __name__ == "__main__":
    unittest.main()
