from collections import Counter
from copy import deepcopy


cdef class _CyVocStrInterface:
    cdef object s2c

    def __init__(self, s2c):
        self.s2c = s2c

    def __contains__(self, str_):
        return str_ in self.s2c


cdef class Vocab:
    cpdef public bint has_unk
    cpdef public str unk
    cpdef public set specials
    cdef public object s2c
    cdef public set _specials_maybe_w_unk
    cpdef public _CyVocStrInterface string

    def __init__(self, specials=None, unk=False):
        if specials is None:
            specials = set()

        if unk is False or unk is None:
            self.unk = ""
            self.has_unk = False
        else:
            self.unk = unk
            self.has_unk = True
        self.specials = specials
        self.s2c = Counter()
        self._specials_maybe_w_unk = deepcopy(self.specials)
        if self.has_unk:
            self._specials_maybe_w_unk.add(self.unk)
        for word in self._specials_maybe_w_unk:
            self.s2c[word] = float("inf")
        self.string = _CyVocStrInterface(self.s2c)

    def __len__(self):
        return len(self.s2c)

    def add(self, word):
        if word in self.s2c:
            self.s2c[word] += 1
        else:
            self.s2c[word] = 1

    def add_iterable(self, words):
        for word in words:
            self.add(word)

    def uncount(self, word):
        self.s2c[word] -= 1
        if self.s2c[word] == 0:
            del self.s2c[word]

    def uncount_iterable(self, words):
        for word in words:
            self.uncount(word)

    def strip(self, n_to_keep=float("inf"), min_freq=0, minimal=True):
        n_to_keep += len(self._specials_maybe_w_unk)
        if n_to_keep < len(self.s2c):
            s2c_n = self.s2c.most_common(n_to_keep)
        else:
            s2c_n = [(s, c) for s, c in self.s2c.items()]

        s2n_f = {s: c for s, c in self.s2c.items() if c >= min_freq}
        if minimal:
            self.s2c = Counter({s: c for s, c in s2c_n if s in s2n_f})
        else:
            s2c_n = Counter(s2c_n)
            s2c_n.update(s2n_f)
            self.s2c = s2c_n
        self.string = _CyVocStrInterface(self.s2c)
