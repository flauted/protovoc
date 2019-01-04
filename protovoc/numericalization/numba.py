import numba
import numpy as np
import bisect


# num .str
class _NbStrInterface:
    def __init__(self, i2s):
        self.i2s = i2s

    def __getitem__(self, integer):
        try:
            return self.i2s[integer]
        except:
            return [self.i2s[i] for i in integer]

    def __contains__(self, str_):
        return str_ in self.i2s


class _NbIntInterface:
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


@numba.njit('b1[:](i8[:],i8,i8)')
def _isin_cumsum_ge0_stepped(arr, thresh, stepsize):
    n_elems = arr.size
    mask = np.empty(n_elems, dtype=np.bool_)
    for group_idx in range(0, (n_elems / stepsize)):
        for elem_idx in range(group_idx * stepsize, (group_idx + 1) * stepsize):
            if arr[elem_idx] < thresh:
                mask[elem_idx:stepsize*(group_idx+1)] = True
                break
            mask[elem_idx] = False
    return mask


@numba.jit()
def le_or_after(arr, thresh, axis):
    stepsize = arr.shape[axis]
    if axis != arr.ndim - 1:
        # tp_axes = np.arange(arr.ndim)
        # tp_axes[axis], tp_axes[-1] = tp_axes[-1], tp_axes[axis]
        # arr = np.transpose(arr, axes=tp_axes)
        arr = np.swapaxes(arr, -1, axis)
        return _isin_cumsum_ge0_stepped(np.ravel(arr), thresh, stepsize).reshape(
            arr.shape).swapaxes(-1, axis)  # .transpose(np.argsort(tp_axes))
    else:
        return _isin_cumsum_ge0_stepped(np.ravel(arr), thresh, stepsize).reshape(
            arr.shape)


class Numericalization:
    def __init__(self, vocab):
        self.specials = vocab.specials
        self.unk = vocab.unk
        self.has_unk = self.unk is not False and self.unk is not None
        self.specials_w_unk = vocab.specials_w_unk
        self._n_spec = len(self.specials_w_unk)
        unordered_cts = np.asarray(list(vocab.s2c.values()), dtype=np.float64)
        idxs_desc = np.argsort(unordered_cts)[::-1]
        unordered_strs = np.asarray(list(vocab.s2c.keys()), dtype=object)
        self.cts = unordered_cts[idxs_desc]
        self.i2s = unordered_strs[idxs_desc]
        self.s2i = {s: i for i, s in enumerate(self.i2s)}
        unk_interface = False if not self.unk else self.s2i[self.unk]
        self.integer = _NbIntInterface(self.s2i, unk_interface)
        self.permit_unk(False)
        self.string = _NbStrInterface(self.i2s)

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

    def sentence(self, integers, axis=0):
        strs = self.i2s[integers]
        idx = tuple([slice(0, s) if i != axis else slice(1, s) for i, s in enumerate(strs.shape)])
        strs[idx] = " " + strs[idx]
        strs[le_or_after(integers, self._cutoff, axis)] = ""
        strs = strs.sum(axis=axis)
        return strs

    def permit_unk(self, val):
        if val:
            self._cutoff = self._n_spec - 1
        else:
            self._cutoff = self._n_spec

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
        self.integer = _NbIntInterface(self.s2i, unk_idx)
        self.string = _NbStrInterface(self.i2s)
