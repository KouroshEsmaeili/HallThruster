import numpy as np


class Tridiagonal:

    def __init__(self, dl, d, du):
        dl = np.array(dl,dtype=np.float64)
        d = np.array(d,dtype=np.float64)
        du = np.array(du,dtype=np.float64)

        n_main = d.shape[0]
        n_sub = dl.shape[0]
        n_super = du.shape[0]

        if n_sub != n_main - 1 or n_super != n_main - 1:
            raise ValueError("Lengths mismatch: sub & super diag must be (n-1) if main diag is n.")

        self._dl = dl
        self._d = d
        self._du = du
        self._n = n_main

    @property
    def n(self):
        return self._n

    def shape(self):
        return (self._n, self._n)

    def to_dense(self):
        mat = np.zeros((self._n, self._n), dtype=float)
        # fill main diagonal
        for i in range(self._n):
            mat[i, i] = self._d[i]
        # fill subdiagonal
        for i in range(self._n - 1):
            mat[i + 1, i] = self._dl[i]
        # fill superdiagonal
        for i in range(self._n - 1):
            mat[i, i + 1] = self._du[i]
        return mat

    def matvec(self, x):
        x = np.array(x,dtype=np.float64)
        if x.shape[0] != self._n:
            raise ValueError(f"Size mismatch: x must have length {self._n}")

        y = np.zeros_like(x)
        # y[0] = d[0]*x[0] + du[0]*x[1]
        # y[i] = dl[i-1]* x[i-1] + d[i]* x[i] + du[i]* x[i+1] (for i in [1..n-2])
        # y[n-1] = dl[n-2]* x[n-2] + d[n-1]* x[n-1]

        # first row
        y[0] = self._d[0] * x[0] + self._du[0] * x[1] if self._n > 1 else self._d[0] * x[0]

        # middle rows
        for i in range(1, self._n - 1):
            y[i] = (self._dl[i - 1] * x[i - 1] +
                    self._d[i] * x[i] +
                    self._du[i] * x[i + 1])

        # last row
        if self._n > 1:
            y[self._n - 1] = (self._dl[self._n - 2] * x[self._n - 2] +
                              self._d[self._n - 1] * x[self._n - 1])

        return y

    def solve(self, b):
        b = np.array(b,dtype=np.float64)
        if b.shape[0] != self._n:
            raise ValueError(f"Size mismatch: b must have length {self._n}")

        # We copy the diagonals because the Thomas algorithm modifies them in place
        dl = np.copy(self._dl)
        d = np.copy(self._d)
        du = np.copy(self._du)

        # Forward sweep
        for i in range(1, self._n):
            w = dl[i - 1] / d[i - 1]
            d[i] = d[i] - w * du[i - 1]
            b[i] = b[i] - w * b[i - 1]

        # back-substitution
        x = np.zeros_like(b)
        x[-1] = b[-1] / d[-1]
        for i in range(self._n - 2, -1, -1):
            x[i] = (b[i] - du[i] * x[i + 1]) / d[i]

        return x

    def __repr__(self):
        return f"<Tridiagonal n={self._n} sub={self._dl} main={self._d} super={self._du}>"



# Example usage:
if __name__ == "__main__":
    a = Tridiagonal(np.ones(10), np.ones(11), np.ones(10))
    print(a)
    # small test
    tri = Tridiagonal([1, 1], [4, 4, 4], [2, 2])
    print("shape:", tri.shape())
    dense = tri.to_dense()
    print("Dense:\n", dense)
    x = np.array([1.0, 2.0, 3.0],dtype=np.float64)
    y = tri.matvec(x)
    print("matvec x =>", y)

    # Solve T*x=b
    b = np.array([7.0, 14.0, 23.0],dtype=np.float64)
    sol_x = tri.solve(b)
    print("solution x =>", sol_x)
    print("Check T*x =>", tri.matvec(sol_x))


def tridiagonal_forward_sweep_inplace(A: Tridiagonal, b: np.ndarray):
    n = A.n
    # Loop from the second row (i = 1 in 0-based indexing) to n-1.
    for i in range(1, n):
        w = A._dl[i - 1] / A._d[i - 1]
        A._d[i] = A._d[i] - w * A._du[i - 1]
        b[i] = b[i] - w * b[i - 1]


def tridiagonal_backward_sweep_inplace(y: np.ndarray, A: Tridiagonal, b: np.ndarray):
    n = A.n
    y[n - 1] = b[n - 1] / A._d[n - 1]
    # Loop backwards from n-2 down to 0.
    for i in range(n - 2, -1, -1):
        y[i] = (b[i] - A._du[i] * y[i + 1]) / A._d[i]


def tridiagonal_solve_inplace(y: np.ndarray, A: Tridiagonal, b: np.ndarray):
    tridiagonal_forward_sweep_inplace(A, b)
    tridiagonal_backward_sweep_inplace(y, A, b)


def tridiagonal_solve(A: Tridiagonal, b: np.ndarray) -> np.ndarray:
    y = np.empty_like(b)
    A_copy = A.copy()
    b_copy = b.copy()
    tridiagonal_solve_inplace(y, A_copy, b_copy)
    return y
