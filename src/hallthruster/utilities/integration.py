
import numpy as np

def cumtrapz(x, y, y0=0.0):
    if len(x) != len(y):
        print('[cumtrapz]: len(x) != len(y)')
    int_arr = np.zeros(len(x), dtype=float)

    cumtrapz_inplace(int_arr, x, y, y0)
    return int_arr

def cumtrapz_inplace(cache, x, y, y0=0.0):
    # print('[cumtrapz_inplace] y0:', y0)

    # Check lengths
    if len(x) != len(y):
        print('[cumtrapz_inplace]: len(x) != len(y)')
    n = len(x)
    if len(y) != n or len(cache) != n:
        raise ValueError("x, y, and cache must all be the same length.")

    cache[0] = y0
    for i in range(1, n):
        dx = x[i] - x[i-1]
        cache[i] = cache[i-1] + 0.5 * dx * (y[i] + y[i-1])
        # print('[cumtrapz_inplace] cache["ϕ"]:', cache)

    return cache

