import numpy as np


class ART1:
    def __init__(self, m=10, k=10, r=0.6):
        self.r = r
        self.m = m
        self.k = k
        self.w = np.zeros(shape=(self.m, self.k), dtype=float)
        self.t = np.zeros(shape=(self.m, self.k), dtype=float)
        self.lam = 1.0
        self.v = 0.5
        self.members = np.zeros(self.k, dtype=float)

    def initialize(self, x):
        assert len(x) == self.m
        sum_x = x.sum()

        assert 0 in self.members, "Too many neurons"

        index = np.nonzero(self.members == 0)[0][0]
        self.members[index] += 1

        print('Create new cluster {} for {}'.format(index, x))

        for i in range(self.m):
            self.w[i, index] = (self.lam * x[i]) / (self.lam - 1 + sum_x)
            # self.w[i, self.number_active] = x[i] / (self.lam + sum_x)
            self.t[i, index] = x[i]
        return index

    def clear(self):
        self.w = np.zeros(shape=(self.m, self.k), dtype=float)
        self.t = np.zeros(shape=(self.m, self.k), dtype=float)

    def _predict(self, x):
        assert len(x) == self.m
        y = np.dot(x, self.w)
        sum_x = x.sum()

        if y.sum() == 0:
            return -1

        args = y.argsort()[::-1]

        for arg in args:
            r_new = np.dot(self.t[:, arg], x) / sum_x
            if r_new > self.r:
                return arg
        return -1

    def recalculation(self, examples, arg):
        for x in examples:
            sum_x = x.sum()
            for i in range(self.m):
                self.w[i, arg] = (1 - self.v) * self.w[i, arg] + self.v * (self.lam * x[i]) / (self.lam - 1 + sum_x)
                # self.w[i, arg] = (1-self.v)*self.w[i, arg] + self.v * x[i] / (self.lam + sum_x)
                self.t[i, arg] = (1 - self.v) * self.t[i, arg] + self.v * x[i]

    def fit_transform(self, features):
        assert features.ndim == 2
        self.m = features.shape[1]
        self.clear()

        clusters = np.ones(features.shape[0], dtype=int) * -1
        done = True

        while done:
            done = False

            for i, x in enumerate(features):
                cluster = self._predict(x)

                # if cluster not found, create new
                if cluster == -1:
                    if clusters[i] != -1:
                        self.members[clusters[i]] -= 1
                    clusters[i] = self.initialize(x)

                    continue

                if cluster != clusters[i]:
                    done = True

                    self.members[cluster] += 1
                    self.recalculation(x.reshape(1, self.m), cluster)

                    old_cluster = clusters[i]
                    clusters[i] = cluster
                    if old_cluster != -1:
                        self.members[old_cluster] -= 1
                        self.recalculation(features[clusters == old_cluster], old_cluster)

        for i in range(clusters.max() + 1):
            print(i)
            print(features[clusters == i])
        print()

        return np.array(clusters)


if __name__ == '__main__':
    '''
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    handler_log = logging.FileHandler('log.log')
    handler_log.setLevel(logging.INFO)
    formatter_log = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_log.setFormatter(formatter_log)
    log.addHandler(handler_log)
    '''

    art = ART1()
    database = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                         [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                         [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])

    print(art.fit_transform(database))
