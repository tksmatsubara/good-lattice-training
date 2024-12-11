import numpy as np
from scipy.stats.qmc import Sobol
from pyDOE import lhs

GLT_table = {
    2: {
        3: [1, 2],
        5: [1, 3],
        8: [1, 5],
        13: [1, 8],
        21: [1, 13],
        34: [1, 21],
        55: [1, 34],
        89: [1, 55],
        144: [1, 89],
        233: [1, 144],
        377: [1, 233],
        610: [1, 377],
        987: [1, 610],
        1597: [1, 987],
        2584: [1, 1597],
        4181: [1, 2584],
        6765: [1, 4181],
        10946: [1, 6765],
        17711: [1, 10946],
        28657: [1, 17711],
        46368: [1, 28657],
        75025: [1, 46368],
        121393: [1, 75025],
        196418: [1, 121393],
    },
    4: {
        60: [1, 8, 18, 22],
        118: [1, 18, 40, 52],
        180: [1, 8, 46, 74],
        286: [1, 16, 94, 138],
        440: [1, 21, 136, 216],
        307: [1, 42, 229, 101],
        562: [1, 53, 89, 221],
        701: [1, 82, 415, 382],
        1019: [1, 71, 765, 865],
        2129: [1, 776, 1281, 1906],
        3001: [1, 174, 266, 1269],
        4001: [1, 113, 766, 2537],
        5003: [1, 792, 1889, 191],
        6007: [1, 1351, 5080, 3086],
        8191: [1, 2488, 5939, 7859],
        10007: [1, 1206, 3421, 2842],
        20039: [1, 19668, 17407, 14600],
    },
    5: None,
    6: {
        2129: [1, 41, 1681, 793, 578, 279],
        3001: [1, 233, 271, 122, 1417, 51],
        4001: [1, 1751, 1235, 1945, 844, 1475],
        5003: [1, 2037, 1882, 1336, 4803, 2846],
        6007: [1, 312, 1232, 5943, 4060, 5250],
        8191: [1, 1632, 1349, 6380, 1399, 6070],
        10007: [1, 2240, 4093, 1908, 931, 3984],
        15019: [1, 8743, 8358, 6559, 2795, 772],
        20039: [1, 5557, 150, 11951, 2461, 9179],
        33139: [1, 18236, 1831, 19143, 5522, 22910],
        51097: [1, 9931, 7551, 29682, 44446, 17340],
        71053: [1, 18010, 3155, 50203, 6605, 13328],
    },
    7: None,
    8: {
        3997: [1, 3888, 3564, 3034, 2311, 1417, 375, 3211],
        11215: [1, 10909, 10000, 8512, 6485, 3976, 1053, 9010],
        24041: [1, 17441, 21749, 5411, 12326, 3144, 21024, 6252],
        28832: [1, 27850, 24938, 20195, 13782, 5918, 25703, 15781],
        33139: [1, 3520, 29553, 3239, 1464, 16735, 19197, 3019],
        46213: [1, 5347, 30775, 35645, 11403, 16894, 32016, 16600],
        57091: [1, 17411, 46802, 9779, 16807, 35302, 1416, 47755],
    },
}


class GLTsD:
    sampler = None
    roots_legendre = {}

    @staticmethod
    def get_lattice(Nf, s, gs):
        gs = np.random.permutation(gs)
        ss = np.random.uniform(0, Nf, s)
        X_0 = ((np.arange(Nf)[:, np.newaxis] * gs + ss) % Nf) / Nf
        return X_0, np.ones(Nf)

    @staticmethod
    def get_evaluation_points(lattice, s, lb, ub, Nf):
        if lattice == "lhs":
            X_0 = lhs(s, Nf)  # Nf x s, Nf points in s-dimendional space (bounded by lb and ub)
            f_w = np.ones(Nf)
        elif lattice == "uni":
            X_0 = np.random.uniform(0.0, 1.0, (Nf, s))  # Nf x s, Nf points in s-dimendional space (bounded by lb and ub)
            f_w = np.ones(Nf)
        elif lattice == "sq":
            ns = int(Nf ** (1 / s) + 0.5)
            assert Nf == ns**s
            Xs = np.meshgrid(*([np.arange(0, 1, 1 / ns) + np.random.uniform(0, 1 / ns)] * s))
            X_0 = np.hstack([X.flatten()[:, None] for X in Xs])
            f_w = np.ones(Nf)
        elif lattice == "sob":
            if GLTsD.sampler is None:
                GLTsD.sampler = Sobol(d=s, scramble=True)
            try:
                X_0 = GLTsD.sampler.random(n=Nf)
            except:
                GLTsD.sampler.reset()
                X_0 = GLTsD.sampler.random(n=Nf)
            f_w = np.ones(Nf)
        elif lattice == "glt":
            X_0, f_w = GLTsD.get_lattice(Nf, s, GLT_table[s][Nf])
        elif lattice == "gau":
            ns = int(Nf ** (1 / s) + 0.5)
            assert Nf == ns**s
            if ns not in GLTsD.roots_legendre:
                import scipy.special

                x, w = scipy.special.roots_legendre(ns)
                x = (x + 1) / 2  # [0,1]
                Xs = np.meshgrid(*([x] * s))
                X_0 = np.hstack([X.flatten()[:, None] for X in Xs])
                f_w = np.ones(1)
                for i in range(s):
                    w_ = w.copy()
                    for j in range(s):
                        if j < i:
                            w_ = np.expand_dims(w_, 0)
                        if j > i:
                            w_ = np.expand_dims(w_, -1)
                    f_w = f_w * w_
                f_w = f_w.flatten() / f_w.mean()
                GLTsD.roots_legendre[ns] = (X_0, f_w)
            X_0, f_w = GLTsD.roots_legendre[ns]
        else:
            raise NotImplementedError("invalid lattice:", lattice)
        X_f = lb + (ub - lb) * X_0

        return X_f, f_w
