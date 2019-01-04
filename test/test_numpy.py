import unittest

from test.general_tests import NumericalizationTestSuite
from protovoc.numericalization.numpy import Numericalization
from protovoc.vocab.basic import Vocab


class TestNumpyNumericalization(unittest.TestCase, NumericalizationTestSuite):
    _voc = Vocab
    _num = Numericalization


if __name__ == "__main__":
    unittest.main()
