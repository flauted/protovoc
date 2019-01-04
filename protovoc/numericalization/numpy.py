import numpy as np
from copy import deepcopy
from collections import Counter
import bisect


# num .str
class _NpStrInterface:
    def __init__(self, i2s):
        self.i2s = i2s

    def __getitem__(self, integer):
        try:
            return self.i2s[integer]
        except:
            return [self.i2s[i] for i in integer]

    def __contains__(self, str_):
        return str_ in self.i2s


class _NpIntInterface:
    def __init__(self, s2i, unk):
        self.s2i = s2i
        if unk is not False:
            self.unk = unk

    def __getitem__(self, string):
        if isinstance(string, str):
            try:
                return self.s2i[string]
            except KeyError:
                try:
                    return self.unk
                except AttributeError:
                    raise KeyError(f"Couldn't find {string}")
        else:
            return [self[s] for s in string]

    def __contains__(self, int_):
        return int_ < len(self.s2i)


class Numericalization:
    def __init__(self, vocab):
        self.specials = vocab.specials
        self.unk = vocab.unk
        self.specials_w_unk = vocab.specials_w_unk
        unordered_cts = np.asarray(list(vocab.s2c.values()), dtype=np.float64)
        idxs_desc = np.argsort(unordered_cts)[::-1]
        unordered_strs = np.asarray(list(vocab.s2c.keys()), dtype=object)
        self.cts = unordered_cts[idxs_desc]
        self.i2s = unordered_strs[idxs_desc]
        self.s2i = {s: i for i, s in enumerate(self.i2s)}
        unk_interface = False if not self.unk else self.s2i[self.unk]
        self.integer = _NpIntInterface(self.s2i, unk_interface)
        self.specs_as_int = np.asarray(self.integer[self.specials], dtype=np.int64)
        self.specs_w_unk_as_int = np.asarray(self.integer[self.specials_w_unk], dtype=np.int64)
        self.permit_unk(False)
        self.string = _NpStrInterface(self.i2s)

    def sentence(self, integers, axis=0):
        sucks = np.isin(integers, self.spec_ints).cumsum(axis=axis) != 0
        strs = self.i2s[integers]
        idx = tuple([slice(0, s) if i != axis else slice(1, s) for i, s in enumerate(strs.shape)])
        strs[idx] = " " + strs[idx]
        strs[sucks] = ""
        strs = strs.sum(axis=axis)
        return strs

    def permit_unk(self, val):
        if val:
            self.spec_ints = self.specs_as_int
        else:
            self.spec_ints = self.specs_w_unk_as_int

    def __len__(self):
        return len(self.cts)

    def strip(self, n_to_keep=float("inf"), min_freq=0, minimal=True):
        if min_freq > 0:
            n_freq_enough = len(self.cts) - bisect.bisect_left(self.cts[::-1], min_freq)
        else:
            n_freq_enough = len(self.cts)

        n_to_keep += len(self.specials_w_unk)

        if minimal:
            n_to_keep = min(n_freq_enough, n_to_keep)
        else:
            n_to_keep = max(n_freq_enough, n_to_keep)
        if n_to_keep >= len(self.cts):
            return
        self.cts = self.cts[:n_to_keep]
        self.i2s = self.i2s[:n_to_keep]
        self.s2i = {s: i for i, s in enumerate(self.i2s)}
        if self.unk:
            unk_idx = self.s2i[self.unk]
        else:
            unk_idx = False
        self.integer = _NpIntInterface(self.s2i, unk_idx)
        self.string = _NpStrInterface(self.i2s)
