import numpy as np


class ART1:
    def __init__(self, m, k, r):
        self.r = r
        self.m = m
        self.k = k
        self.w = np.zeros(shape=(m, k), dtype=float)
        self.t = np.zeros(shape=(m, k), dtype=float)
        self.lam = 2.0
        self.v = 0.5
        self.number_active = -1

    def initialize(self, x):
        assert len(x) == self.m
        sum_x = x.sum()
        self.number_active += 1

        for i in range(self.m):
            self.w[i, self.number_active] = (self.lam * x[i]) / (self.lam - 1 + sum_x)
            self.t[i, self.number_active] = x[i]

    def recognize(self, x):
        assert len(x) == self.m
        y = np.dot(x, self.w)
        sum_x = x.sum()

        args = y.argsort()[::-1]

        winner = False

        for arg in args:
            r_new = np.dot(self.t[:, arg], x) / sum_x
            if r_new > self.r:
                winner = True
                print(arg)
                for i in range(self.m):
                    self.w[i, args] = (1-self.v)*self.w[i, args] + self.v * (self.lam * x[i]) / (self.lam - 1 + sum_x)
                    self.t[i, args] = (1-self.v)*self.t[i, args] + self.v * x[i]

        if not winner:
            self.initialize(x)


if __name__ == '__main__':
    art = ART1(5, 3, 0.5)
    art.initialize(np.array([0, 1, 0, 1, 0]))
    art.recognize(np.array([0, 1, 0, 1, 0]))
    art.recognize(np.array([0, 1, 1, 0, 0]))
    art.recognize(np.array([0, 1, 1, 0, 0]))
