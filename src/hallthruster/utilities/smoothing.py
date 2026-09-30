import numpy as np
from ..utilities.interpolation import lerp


def linear_transition(x, cutoff, L, y1, y2):
    x1 = cutoff - L / 2.0
    x2 = cutoff + L / 2.0
    if x < x1:
        return y1
    elif x > x2:
        return y2
    else:
        return lerp(x, x1, x2, y1, y2)

def smooth_if(x, cutoff, y1, y2, L):
    return ((y2 - y1) * np.tanh((x - cutoff) / (L / 4.0)) + y1 + y2) / 2.0

def smooth(x, x_cache, iters=1):
    n = len(x)
    if iters > 0:
        # Copy x into x_cache
        np.copyto(x_cache, x)
        # In Julia: x_cache[1] = x[2] → in Python: index 0 becomes the second element.
        x_cache[0] = x[1]
        # In Julia: x_cache[end - 1] = x[end] → in Python: set the second-to-last element.
        x_cache[n - 2] = x[n - 1]
        # Loop over indices corresponding to Julia's 2:(length(x)-1)
        for i in range(1, n - 1):
            # For the first and last index in the loop:
            if i == 1 or i == n - 2:
                x_cache[i] = 0.5 * x[i] + 0.25 * (x[i - 1] + x[i + 1])
            else:
                x_cache[i] = (0.4 * x[i] +
                              0.24 * (x[i - 1] + x[i + 1]) +
                              0.06 * (x[i - 2] + x[i + 2]))
        # Copy the smoothed values back into x
        np.copyto(x, x_cache)
        return smooth(x, x_cache, iters=iters - 1)
    else:
        return x

# Example usage:
if __name__ == "__main__":
    # Test linear_transition
    x_val = 5.0
    cutoff = 10.0
    L = 4.0
    y1 = 0.0
    y2 = 1.0
    print("linear_transition:", linear_transition(x_val, cutoff, L, y1, y2))

    # Test smooth_if
    print("smooth_if:", smooth_if(x_val, cutoff, y1, y2, L))

    # Test smooth: create a noisy array and a cache array.
    x = np.linspace(0, 10, 11)
    x_noise = x + np.random.normal(scale=0.1, size=x.shape)
    x_cache = np.empty_like(x_noise)
    print("Before smoothing:", x_noise)
    x_smoothed = smooth(x_noise, x_cache, iters=3)
    print("After smoothing:", x_smoothed)
