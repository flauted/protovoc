import random
import time
import numpy as np


def fake_data(file="rando.txt"):
    # note: inefficient, but who cares?
    with open(file, "r") as f:
        newtxt = []
        txt = f.read()
        for c in txt:
            if c == ".":
                c = " ."
            newtxt.append(c)
    txt = "".join(newtxt).lower()
    txt = txt.replace("\n", "").split(" ")
    return txt


def benchmark_numericalizing(voc, num, n_trials=1000):
    queries = list(range(1000))
    times = []
    # TODO: Numericalization is allowed to be destructive. This isn't necessarily right
    for _ in queries:
        start = time.time()
        _ = num(voc)
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_numericalizing.__name__} ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_numericalizing.__name__, times


def benchmark_single_word_random_access(voc, n_trials=1000):
    queries = [random.randint(0, len(voc)-1) for _ in range(n_trials)]
    times = []
    for query in queries:
        start = time.time()
        _ = voc.string[query]
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_single_word_random_access.__name__} ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_single_word_random_access.__name__, times


def benchmark_1d_word_array_getitem_random_access(voc, n_trials=1000):
    lens = [random.randint(3, 12) for _ in range(n_trials)]
    queries = [[random.randint(0, len(voc)-1) for _ in range(len_)] for len_ in lens]
    times = []
    for query in queries:
        start = time.time()
        _ = voc.string[query]
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_1d_word_array_getitem_random_access.__name__} ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_1d_word_array_getitem_random_access.__name__, times


def benchmark_1d_word_array_sentence_random_access(voc, n_trials=1000):
    seq_len = 20
    queries = [np.random.randint(0, len(voc)-1, (seq_len)) for _ in range(n_trials)]
    times = []
    for query in queries:
        start = time.time()
        _ = voc.sentence(query, axis=0)
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_1d_word_array_sentence_random_access.__name__}"
          f" ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_1d_word_array_sentence_random_access.__name__, times


def benchmark_2d_word_array_getitem_random_access(voc, n_trials=1000):
    batch = 64
    all_lens = [[random.randint(3, 12) for _ in range(batch)] for _ in range(n_trials)]
    queries = [
        [[random.randint(0, len(voc)-1) for _ in range(l)] for l in lens]
        for lens in all_lens]
    times = []
    for query in queries:
        start = time.time()
        _ = voc.string[query]
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_2d_word_array_getitem_random_access.__name__} ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_2d_word_array_getitem_random_access.__name__, times


def benchmark_2d_word_array_sentence_random_access(voc, n_trials=1000):
    seq_len = 20
    batch = 64
    queries = [np.random.randint(0, len(voc) - 1, (seq_len, batch)) for _ in range(n_trials)]
    stops = [np.random.randint(1, seq_len, (batch,)) for _ in range(n_trials)]
    for i in range(n_trials):
        queries[i][..., stops[i]] = 0
    times = []
    for query in queries:
        start = time.time()
        _ = voc.sentence(query, axis=0)
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_2d_word_array_sentence_random_access.__name__}"
          f" ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_2d_word_array_sentence_random_access.__name__, times


def benchmark_3d_word_array_sentence_random_access(voc, n_trials=1000):
    seq_len = 20
    batch = 64
    beam_size = 10
    queries = [np.random.randint(0, len(voc) - 1, (seq_len, beam_size, batch)) for _ in range(n_trials)]
    stops = [np.random.randint(1, seq_len, (beam_size, batch)) for _ in range(n_trials)]
    for i in range(n_trials):
        queries[i][..., stops[i]] = 0
    times = []
    for query in queries:
        start = time.time()
        _ = voc.sentence(query, axis=0)
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_3d_word_array_sentence_random_access.__name__}"
          f" ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_3d_word_array_sentence_random_access.__name__, times


def benchmark_uncounting_iterable(klass, unk, n_trials=25):
    times = []
    for _ in range(n_trials):
        voc = klass(unk=unk)
        txt = fake_data()
        voc.add_iterable(txt)
        txt_to_rm = fake_data("rando_short.txt")
        start = time.time()
        voc.uncount_iterable(txt_to_rm)
        end = time.time()
        times.append( end - start)
    print(f"{benchmark_uncounting_iterable.__name__} ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_uncounting_iterable.__name__, times


def benchmark_adding_iterable(klass, unk, n_trials=25):
    times = []
    for _ in range(n_trials):
        voc = klass(unk=unk)
        txt = fake_data()
        start = time.time()
        voc.add_iterable(txt)
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_adding_iterable.__name__} ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_adding_iterable.__name__, times


def benchmark_strip_n_words(klass, unk, n_trials=25):
    times = []
    for _ in range(n_trials):
        voc = klass(unk=unk)
        txt = fake_data()
        voc.add_iterable(txt)
        n_to_keep = int(len(voc) * 0.20)
        start = time.time()
        voc.strip(n_to_keep=n_to_keep)
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_strip_n_words.__name__} ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_strip_n_words.__name__, times


def benchmark_strip_n_words_numericalized(klass, num, unk, n_trials=25):
    times = []
    for _ in range(n_trials):
        voc = klass(unk=unk)
        txt = fake_data()
        voc.add_iterable(txt)
        voc = num(voc)
        n_to_keep = int(len(voc) * 0.20)
        start = time.time()
        voc.strip(n_to_keep=n_to_keep)
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_strip_n_words_numericalized.__name__} ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_strip_n_words_numericalized.__name__, times


def benchmark_strip_by_freq(klass, unk, n_trials=25):
    times = []
    for _ in range(n_trials):
        voc = klass(unk=unk)
        txt = fake_data()
        voc.add_iterable(txt)
        start = time.time()
        voc.strip(min_freq=3)
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_strip_by_freq.__name__} ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_strip_by_freq.__name__, times


def benchmark_strip_by_freq_numericalized(klass, num, unk, n_trials=25):
    times = []
    for _ in range(n_trials):
        voc = klass(unk=unk)
        txt = fake_data()
        voc.add_iterable(txt)
        voc = num(voc)
        start = time.time()
        voc.strip(min_freq=3)
        end = time.time()
        times.append(end - start)
    print(f"{benchmark_strip_by_freq_numericalized.__name__} ({n_trials}): {np.mean(times)} s, avg")
    return benchmark_strip_by_freq_numericalized.__name__, times


def benchmarks(klass, num, unk):
    bench = []
    bench.append(benchmark_adding_iterable(klass, unk))
    bench.append(benchmark_uncounting_iterable(klass, unk))
    bench.append(benchmark_strip_n_words(klass, unk))
    bench.append(benchmark_strip_n_words_numericalized(klass, num, unk))
    bench.append(benchmark_strip_by_freq(klass, unk))
    bench.append(benchmark_strip_by_freq_numericalized(klass, num, unk))

    voc = klass(unk=unk)
    # benchmark
    txt = fake_data()
    voc.add_iterable(txt)
    bench.append(benchmark_numericalizing(voc, num))
    voc = num(voc)

    bench.append(benchmark_single_word_random_access(voc))

    # 1D word array random access
    bench.append(benchmark_1d_word_array_getitem_random_access(voc))

    # 1D word array, sentence random access
    bench.append(benchmark_1d_word_array_sentence_random_access(voc))

    # 2D word array, random access
    bench.append(benchmark_2d_word_array_getitem_random_access(voc))

    # 2D word array, sentence random access
    bench.append(benchmark_2d_word_array_sentence_random_access(voc))

    # 3D word array, sentence random access
    bench.append(benchmark_3d_word_array_sentence_random_access(voc))
    return dict(bench)
