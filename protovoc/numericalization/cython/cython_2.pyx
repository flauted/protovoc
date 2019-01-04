import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int64_t LONG_t
ctypedef np.float64_t FLOAT_t
ctypedef np.uint8_t BOOL_t


cdef class _CyStrInterface:
    cdef readonly np.ndarray i2s
    def __init__(self, i2s):
        self.i2s = i2s

    def __getitem__(self, integer):
        try:
            return self.i2s[integer]
        except:
            return [self.i2s[i] for i in integer]

    def __contains__(self, str_):
        return str_ in self.i2s


cdef class _CyIntInterface:
    cdef readonly dict s2i
    cdef readonly int unk_i
    cdef readonly bint has_unk
    def __init__(self, s2i, unk, has_unk):
        SENTINEL = -50
        self.s2i = s2i
        if has_unk:
            self.unk_i = self.s2i[unk]
        else:
            self.unk_i = SENTINEL
        self.has_unk = has_unk

    def __getitem__(self, string):
        if isinstance(string, str):
            try:
                return self.s2i[string]
            except KeyError:
                if self.has_unk:
                    return self.unk_i
                else:
                    raise IndexError(f"Couldn't find {string}")
        else:
            return [self[s] for s in string]

    def __contains__(self, int_):
        return int_ < len(self.s2i)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray[BOOL_t, ndim=1, cast=True] _isin_cumsum_ge0(np.ndarray[LONG_t, ndim=1] elements, int cutoff):
    n_elems = len(elements)
    cdef int elem_idx
    for elem_idx in range(0, n_elems):
        if elements[elem_idx] < cutoff:
            return np.concatenate((np.zeros(elem_idx, dtype=bool), np.ones(n_elems - elem_idx, dtype=bool)))

    return np.zeros(n_elems, dtype=bool)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef str _sentence_1d(
        np.ndarray[LONG_t, ndim=1] integers,
        int cutoff,
        np.ndarray[object, ndim=1] strings):
    cdef np.ndarray[object, ndim=1] strs = strings[integers]
    strs[1:] = " " + strs[1:]
    strs[_isin_cumsum_ge0(integers, cutoff)] = ""
    cdef str string = strs.sum()
    return string


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray _sentence(
        np.ndarray integers,
        int axis,
        int cutoff,
        np.ndarray[object, ndim=1] strings):
    cdef np.ndarray strs = strings[integers]
    cdef list idx_l = []
    cdef Py_ssize_t shape
    cdef int i
    for i in range(0, strs.ndim):
        shape = strs.shape[i]
        if i == axis:
            idx_l.append(slice(1, shape))
            continue
        idx_l.append(slice(0, shape))
    cdef tuple idx_t = tuple(idx_l)
    strs[idx_t] = " " + strs[idx_t]
    strs[(integers < cutoff).cumsum(axis=axis) != 0] = ""
    strs = strs.sum(axis=axis)
    return strs


cdef class Numericalization:
    cpdef readonly str unk
    cpdef readonly bint has_unk
    cpdef readonly set specials
    cdef readonly set _specials_maybe_w_unk
    cpdef readonly _CyStrInterface string
    cpdef readonly _CyIntInterface integer
    cdef readonly np.ndarray cts
    cdef readonly np.ndarray i2s
    cdef readonly dict s2i
    cdef readonly int _len_cts
    cdef readonly int _n_spec
    cdef readonly int _cutoff

    def __init__(self, vocab):
        self.specials = vocab.specials
        self.unk = vocab.unk
        self.has_unk = vocab.has_unk
        self._specials_maybe_w_unk = vocab._specials_maybe_w_unk
        self._n_spec = len(self._specials_maybe_w_unk)
        unordered_cts = np.asarray(list(vocab.s2c.values()), dtype=np.float64)
        idxs_desc = np.argsort(unordered_cts)[::-1]
        unordered_strs = np.asarray(list(vocab.s2c.keys()), dtype=object)
        self.cts = unordered_cts[idxs_desc]
        self.i2s = unordered_strs[idxs_desc]
        self.s2i = {s: i for i, s in enumerate(self.i2s)}
        # move unk to last position so that it gets threshed
        if self.has_unk:
            unk_at = self.s2i[self.unk]
            if unk_at != self._n_spec - 1:
                unk_to = self._n_spec - 1
                self.s2i[self.unk] = unk_to
                was_at_unk_to = self.i2s[unk_to]
                self.s2i[was_at_unk_to] = unk_at
                self.i2s[unk_at], self.i2s[unk_to] = self.i2s[unk_to], self.i2s[unk_at]
                self.cts[unk_at], self.cts[unk_to] = self.cts[unk_to], self.cts[unk_to]

        self.integer = _CyIntInterface(self.s2i, self.unk, self.has_unk)
        self.permit_unk(False)
        self.string = _CyStrInterface(self.i2s)
        self._len_cts = len(self.cts)

    def permit_unk(self, val):
        if val:
            self._cutoff = self._n_spec - 1
        else:
            self._cutoff = self._n_spec

    def sentence(self, integers, axis=0):
        if integers.ndim == 1:
            return _sentence_1d(integers, self._cutoff, self.i2s)
        else:
            return _sentence(integers, axis, self._cutoff, self.i2s)

    def __len__(self):
        return self._len_cts

    def strip(self, n_to_keep=float("inf"), min_freq=0, minimal=True):
        # TODO: This is actually only valid if you haven't been adding...
        if min_freq > 0:
            n_freq_enough = self._len_cts - np.searchsorted(self.cts[::-1], min_freq)
        else:
            n_freq_enough = self._len_cts

        n_to_keep += self._n_spec

        if minimal:
            n_to_keep = min(n_freq_enough, n_to_keep)
        else:
            n_to_keep = max(n_freq_enough, n_to_keep)
        if n_to_keep >= len(self.cts):
            return
        self.cts = self.cts[:n_to_keep]
        self._len_cts = len(self.cts)
        self.i2s = self.i2s[:n_to_keep]
        self.s2i = {s: i for i, s in enumerate(self.i2s)}
        self.integer = _CyIntInterface(self.s2i, self.unk, self.has_unk)
        self.string = _CyStrInterface(self.i2s)
