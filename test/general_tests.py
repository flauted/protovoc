import unittest
import numpy as np


class NumericalizationTestSuite:
    """Tests all numericalization classes should pass.

    Attributes
    ----------
    _voc : Class
        A vocab class.
        * ``__init__`` should have arguments
          - unk : bool
          - specials : set
        * Should define ``__contains__`` / work with ``in``.
        * Should define the following "regular" methods:
          - ``add``
          - ``strip``
          - ``uncount``
    _num : Class
        A numericalization class.
        * ``__init__`` should take an instance of ``klass``.
        * Should define ``__contains__`` / work with ``in``.
        * Should have a ``.string`` and ``.integer`` attribute,
          each with ``__getitem__`` indexing support.
        * Should define the following "regular" methods:
          - ``strip``
          - ``sentence``

    """
    def test_add(self):
        for unk in [False, "UNK"]:
            for numericalize in [True, False]:
                with self.subTest(unk=unk, numericalize=numericalize):
                    voc = self._voc(unk=unk)
                    voc.add("hello")
                    if numericalize:
                        voc = self._num(voc)
                    self.assertIn("hello", voc.string)
                    if unk is not False:
                        self.assertIn(unk, voc.string)
                    else:
                        self.assertNotIn(unk, voc.string)
    
    def test_specials_init(self):
        for unk in [False, "UNK"]:
            for numericalize in [True, False]:
                with self.subTest(unk=unk, numericalize=numericalize):
                    specs = {"<pad>", "<bos>", "<eos>"}
                    voc = self._voc(specials=specs, unk=unk)
                    if numericalize:
                        voc = self._num(voc)
                    for spec in specs:
                        self.assertIn(spec, voc.string)
                    if unk is not False:
                        self.assertIn(unk, voc.string)
                    else:
                        self.assertNotIn(unk, voc.string)
    
    def test_strip_min_freq(self):
        for numericalize in [True, False]:
            for unk in ["UNK", False]:
                all_words = {"one", "two", "three", "three_again", "four"}
                for freq, *words in [
                        (0, "one", "two", "three", "three_again", "four"),
                        (1, "one", "two", "three", "three_again", "four"),
                        (2, "two", "three", "three_again", "four"),
                        (3, "three", "three_again", "four"),
                        (4, "four")]:
                    with self.subTest(unk=unk, numericalize=numericalize, freq=freq, words=words):
                        voc = self._voc(unk=unk)
                        for _ in range(3):
                            voc.add("three")
                        for _ in range(2):
                            voc.add("two")
                        for _ in range(1):
                            voc.add("one")
                        for _ in range(3):
                            voc.add("three_again")
                        for _ in range(4):
                            voc.add("four")
        
                        if numericalize:
                            voc = self._num(voc)
        
                        voc.strip(min_freq=freq)
                        if unk:
                            self.assertIn(unk, voc.string)
                        for word in words:
                            self.assertIn(word, voc.string, msg=f"word={word} failed")
                        for word in all_words:
                            if word not in words:
                                self.assertNotIn(word, voc.string, msg=f"word={word} failed")
    
    def test_strip_by_n(self):
        for numericalize in [True, False]:
            for unk in ["UNK", False]:
                all_words = {"one", "two", "three", "three_again", "four"}
                for n_words, *words in [
                        (5, "one", "two", "three", "three_again", "four"),
                        (4, "four", "three", "three_again", "two"),
                        (3, "four", "three", "three_again"),
                        (1, "four")]:
                    with self.subTest(unk=unk, numericalize=numericalize, freq=n_words, words=words):
                        voc = self._voc(unk=unk)
                        for _ in range(3):
                            voc.add("three")
                        for _ in range(2):
                            voc.add("two")
                        for _ in range(1):
                            voc.add("one")
                        for _ in range(3):
                            voc.add("three_again")
                        for _ in range(4):
                            voc.add("four")
        
                        if numericalize:
                            voc = self._num(voc)
        
                        voc.strip(n_to_keep=n_words)
                        if unk:
                            self.assertIn(unk, voc.string)
                        for word in words:
                            self.assertIn(word, voc.string, msg=f"word={word} failed")
                        for word in all_words:
                            if word not in words:
                                self.assertNotIn(word, voc.string, msg=f"word={word} failed")
    
    def test_num_identity(self):
        for unk in [False, "UNK"]:
            voc = self._voc(unk=unk)
            voc.add("hello")
            voc = self._num(voc)
            self.assertEqual("hello", voc.string[voc.integer["hello"]])
    
    def test_num_identity_unk(self):
        voc = self._voc(unk="UNK")
        voc = self._num(voc)
        self.assertEqual("UNK", voc.string[voc.integer["jambalaya"]])
    
    def test_out_of_vocab_no_unk(self):
        voc = self._voc(unk=False)
        voc = self._num(voc)
        with self.assertRaises(Exception):
            voc.string[voc.integer["jambalaya"]]
    
    def test_sentence_1d(self):
        fake_data = np.asarray(
            [2, 3, 4, 1, 2]
        )
        expected = np.asarray([
            "two three four"
        ])
        voc = self._voc(specials={"one"}, unk="UNK")
        for _ in range(5):
            voc.add("two")
        for _ in range(4):
            voc.add("three")
        for _ in range(3):
            voc.add("four")
        voc = self._num(voc)
        s = voc.sentence(fake_data)
        self.assertTrue((expected == s).all())
    
    def test_sentence_1d_unk(self):
        fake_data = np.asarray(
            [2, 3, 4, 0, 0]
        )
        expected = np.asarray([
            "two three four UNK UNK"
        ])
        voc = self._voc(unk="UNK")
        for _ in range(6):
            voc.add("one")
        for _ in range(5):
            voc.add("two")
        for _ in range(4):
            voc.add("three")
        for _ in range(3):
            voc.add("four")
        voc = self._num(voc)
        voc.permit_unk(True)
        s = voc.sentence(fake_data)
        self.assertTrue((expected == s).all())
    
    def test_sentence_2d_axis_0(self):
        fake_data = np.asarray(
            [[2, 3, 4, 1, 2],
             [3, 2, 4, 2, 1],
             [2, 3, 0, 0, 0]]
        )
        expected = np.asarray([
            "two three two", "three two three", "four four", "", "two"
        ])
        voc = self._voc(specials={"one"}, unk="UNK")
        for _ in range(5):
            voc.add("two")
        for _ in range(4):
            voc.add("three")
        for _ in range(3):
            voc.add("four")
        voc = self._num(voc)
        s = voc.sentence(fake_data, axis=0)
        self.assertTrue((expected == s).all())
    
    def test_sentence_2d_axis_1(self):
        fake_data = np.asarray(
            [[2, 3, 4, 1, 2],
             [3, 2, 4, 2, 1],
             [2, 3, 0, 0, 0]]
        )
        expected = np.asarray([
            "two three four", "three two four two", "two three"
        ])
        voc = self._voc(specials={"one"}, unk="UNK")
        for _ in range(5):
            voc.add("two")
        for _ in range(4):
            voc.add("three")
        for _ in range(3):
            voc.add("four")
        voc = self._num(voc)
        s = voc.sentence(fake_data, axis=1)
        self.assertTrue((expected == s).all())
    
    def test_sentence_3d_axis_2(self):
        fake_data = np.asarray(
            [[[2, 3, 4, 1, 2],
              [3, 2, 4, 2, 1],
              [2, 3, 0, 0, 0]],
             [[1, 0, 0, 0, 0],
              [3, 4, 2, 3, 4],
              [4, 4, 3, 3, 0]]]
        )
        expected = np.asarray([
            ["two three four", "three two four two", "two three"],
            ["", "three four two three four", "four four three three"]
        ])
        voc = self._voc(specials={"one"}, unk="UNK")
        for _ in range(5):
            voc.add("two")
        for _ in range(4):
            voc.add("three")
        for _ in range(3):
            voc.add("four")
        voc = self._num(voc)
        s = voc.sentence(fake_data, axis=2)
        self.assertTrue((expected == s).all())
    
    def test_uncount_and_strip(self):
        for unk in [False, "UNK"]:
            with self.subTest(unk=unk):
                voc = self._voc(unk=unk)
                for _ in range(3):
                    voc.add("one")
                for _ in range(5):
                    voc.add("two")
                for _ in range(3):
                    voc.uncount("two")
                voc.strip(min_freq=3)
                self.assertIn("one", voc.string)
                self.assertNotIn("two", voc.string)
    
    def test_uncount_to_death(self):
        for unk in [False, "UNK"]:
            with self.subTest(unk=unk):
                voc = self._voc(unk=unk)
                for _ in range(3):
                    voc.add("one")
                for _ in range(5):
                    voc.add("two")
                for _ in range(5):
                    voc.uncount("two")
                self.assertIn("one", voc.string)
                self.assertNotIn("two", voc.string)
