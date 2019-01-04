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


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray[BOOL_t, ndim=1, cast=True] _isin_cumsum_ge0(np.ndarray[LONG_t, ndim=1] elements, set test_elements):
    cdef int n_elems = len(elements)
    cdef int elem_idx
    for elem_idx in range(0, n_elems):
        if elements[elem_idx] in test_elements:
            return np.concatenate((np.zeros(elem_idx, dtype=bool), np.ones(n_elems - elem_idx, dtype=bool)))

    return np.zeros(n_elems, dtype=bool)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray _isin_cumsum_ge0_general(np.ndarray elements, np.ndarray test_elements, int axis):
    cdef np.ndarray [LONG_t, ndim=1] elements_1 = elements.ravel()
    mask = np.zeros(len(elements_1), dtype=bool)
    cdef int test_idx
    for test_idx in range(0, len(test_elements)):
        mask |= (elements_1 == test_elements[test_idx])
    return mask.reshape([elements.shape[i] for i in range(elements.ndim)]).cumsum(axis=axis) != 0


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
cdef str _sentence_1d(
        np.ndarray[LONG_t, ndim=1] integers,
        # np.ndarray[LONG_t, ndim=1] specs_as_int,
        set specs_as_int,
        np.ndarray[object, ndim=1] strings):
    # cdef np.ndarray sucks = _isin(integers, specs_as_int).cumsum() != 0
    # cdef np.ndarray sucks = _isin_cumsum_ge0(integers, specs_as_int)
    cdef np.ndarray[object, ndim=1] strs = strings[integers]
    strs[1:] = " " + strs[1:]
    strs[_isin_cumsum_ge0(integers, specs_as_int)] = ""
    cdef str string = strs.sum()
    return string


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray _sentence(
        np.ndarray integers,
        int axis,
        np.ndarray[LONG_t, ndim=1] specs_as_int,
        # set specs_as_int,
        np.ndarray[object, ndim=1] strings):
#     cdef np.ndarray sucks = np.isin(integers, specs_as_int).cumsum(axis=axis) != 0
#     cdef np.ndarray sucks =
#     cdef np.ndarray sucks = _isin_cumsum_ge0_general(integers, specs_as_int, axis=axis)
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
    strs[_isin_cumsum_ge0_general(integers, specs_as_int, axis)] = ""
    strs = strs.sum(axis=axis)
    return strs


cdef class Numericalization:
    cpdef readonly str unk
    cpdef readonly bint has_unk
    cpdef readonly set specials
    cdef readonly set _specials_maybe_w_unk
    cdef readonly np.ndarray _specs_as_int
    cdef readonly np.ndarray _specs_maybe_w_unk_as_int
    cdef readonly set _chosen_specs_as_int_set
    cpdef readonly _CyStrInterface string
    cpdef readonly _CyIntInterface integer
    cdef readonly np.ndarray cts
    cdef readonly np.ndarray i2s
    cdef readonly dict s2i
    cdef readonly int _len_cts
    cdef readonly np.ndarray _chosen_specs_as_int

    def __init__(self, vocab):
        self.specials = vocab.specials
        self.unk = vocab.unk
        self.has_unk = vocab.has_unk
        self._specials_maybe_w_unk = vocab._specials_maybe_w_unk
        unordered_cts = np.asarray(list(vocab.s2c.values()), dtype=np.float64)
        idxs_desc = np.argsort(unordered_cts)[::-1]
        unordered_strs = np.asarray(list(vocab.s2c.keys()), dtype=object)
        self.cts = unordered_cts[idxs_desc]
        self.i2s = unordered_strs[idxs_desc]
        self.s2i = {s: i for i, s in enumerate(self.i2s)}
        self.integer = _CyIntInterface(self.s2i, self.unk, self.has_unk)
        self._specs_as_int = np.asarray(self.integer[self.specials], dtype=np.int64)
        self._specs_maybe_w_unk_as_int = np.asarray(self.integer[self._specials_maybe_w_unk], dtype=np.int64)
        self.permit_unk(False)
        self.string = _CyStrInterface(self.i2s)
        self._len_cts = len(self.cts)

    def permit_unk(self, val):
        if val:
            self._chosen_specs_as_int = self._specs_as_int
        else:
            self._chosen_specs_as_int = self._specs_maybe_w_unk_as_int
        self._chosen_specs_as_int_set = set(self._chosen_specs_as_int)

    def sentence(self, integers, axis=0):
        if integers.ndim == 1:
            # return _sentence_1d(integers, self._chosen_specs_as_int, self.i2s)
            return _sentence_1d(integers, self._chosen_specs_as_int_set, self.i2s)
        else:
            return _sentence(integers, axis, self._chosen_specs_as_int, self.i2s)
            # return _sentence(integers, axis, self._chosen_specs_as_int_set, self.i2s)

    def __len__(self):
        return self._len_cts

    def strip(self, n_to_keep=float("inf"), min_freq=0, minimal=True):
        # TODO: This is actually only valid if you haven't been adding...
        if min_freq > 0:
            n_freq_enough = self._len_cts - np.searchsorted(self.cts[::-1], min_freq)
        else:
            n_freq_enough = self._len_cts

        n_to_keep += len(self._specials_maybe_w_unk)

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
