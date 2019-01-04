import unittest

from test.general_tests import NumericalizationTestSuite
from protovoc.numericalization.cython.cython_0 import Numericalization
from protovoc.vocab.basic import Vocab


class TestCython0Numericalization(unittest.TestCase, NumericalizationTestSuite):
    _voc = Vocab
    _num = Numericalization


if __name__ == "__main__":
    unittest.main()
