import pickle
import os
from functools import wraps


def timer(func):
    @wraps(func)
    def inner(*args, **kwargs):
        import time

        start = time.time()
        val = func(*args, **kwargs)
        elapsed = time.time() - start
        print("{} evaluated in {}".format(func.__name__, elapsed))
        return val

    return inner


def update_pickle(key, data, overwrite=False):
    base_dir = os.path.split(__file__)[0]
    pickle_file = os.path.join(base_dir, "data", "peak_data.pickle")
    with open(pickle_file, "rb") as f:
        p = pickle.load(f)
    if key in p and not overwrite:
        raise Exception(
            "{} already exists. Pass overwrite=True to overwrite existing data.".format(
                key
            )
        )
    p[key] = data
    with open(pickle_file, "wb") as o:
        pickle.dump(p, o)
