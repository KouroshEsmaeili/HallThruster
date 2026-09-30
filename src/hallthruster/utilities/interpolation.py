import numpy as np

def lerp(x, x0, x1, y0, y1):
    t = (x - x0) / (x1 - x0)
    return t * (y1 - y0) + y0

def find_left_index(value, array):
    left = -1
    right = len(array)
    while right - left > 1:
        mid = (left + right) // 2
        # print("hellllloiwwwwww",array[mid])
        # print("hellllloiwwwwww",value)
        if array[mid] > value:
            right = mid
        else:
            left = mid
    return left

def interpolate(x, xs, ys, use_log=False):
    i = find_left_index(x, xs)
    if i < 0:
        return ys[0]
    if i >= len(xs) - 1:
        return ys[-1]
    if use_log:
        return np.exp(lerp(x, xs[i], xs[i + 1], np.log(ys[i]), np.log(ys[i + 1])))
    return lerp(x, xs[i], xs[i + 1], ys[i], ys[i + 1])


class LinearInterpolation:

    def __init__(self, x, y, resample_uniform=False, resample_factor=2):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        if resample_uniform:
            xmin, xmax = np.min(x), np.max(x)
            num = int(len(x) * resample_factor)
            resampled_x = np.linspace(xmin, xmax, num)
            # Evaluate the interpolation function at each resampled x.
            resampled_y = np.array([interpolate(_x, x, y) for _x in resampled_x],dtype=np.float64)
            self.xs = resampled_x
            self.ys = resampled_y
        else:
            self.xs = np.array(x,dtype=np.float64)
            self.ys = np.array(y,dtype=np.float64)

    def __call__(self, x, use_log=False):
        # If x is a scalar, process it directly.
        if np.isscalar(x):
            return interpolate(x, self.xs, self.ys, use_log=use_log)
        # Otherwise, treat x as an array and apply interpolation to each element.
        x_arr = np.atleast_1d(x)
        return np.array([interpolate(xi, self.xs, self.ys, use_log=use_log) for xi in x_arr],dtype=np.float64)


if __name__ == "__main__":
    xs = np.linspace(0, 10, 11)
    ys = np.sin(xs)

    itp = LinearInterpolation(xs, ys, resample_uniform=True, resample_factor=2)

    x_val = 4.3
    print("Interpolated value at x =", x_val, "is", itp(x_val))
