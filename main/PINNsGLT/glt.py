import numpy as np
from scipy.stats.qmc import Sobol
from pyDOE import lhs


class GLT:
    table = {
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
    }
    sampler = None
    roots_legendre = {}

    @staticmethod
    def get_lattice(Nf, time_reflected=True, space_reflected=True):
        g1, g2 = GLT.table[Nf]
        if np.random.randn(1) < 0.5:
            g1, g2 = g2, g1
        rx, rt = np.random.uniform(0, Nf, 2)
        x = ((np.arange(Nf) * g1 + rx) % Nf) / Nf
        t = ((np.arange(Nf) * g2 + rt) % Nf) / Nf
        if time_reflected:
            t = t * 2
            t[t >= 1] = 2 - t[t >= 1]
        if space_reflected:
            x = x * 2
            x[x >= 1] = 2 - x[x >= 1]
        X_f = np.stack([x, t], axis=-1)
        return X_f

    @staticmethod
    def get_evaluation_points(lattice, lb, ub, Nf, time_reflected=True, space_reflected=True):
        s = 2
        if lattice == "lhs":
            X_0 = lhs(s, Nf)  # Nf x s, Nf points in s-dimendional space (bounded by lb and ub)
            f_w = np.ones(Nf)
        elif lattice == "uni":
            X_0 = np.random.uniform(0.0, 1.0, (Nf, s))  # Nf x s, Nf points in 2-dimendional space (bounded by lb and ub)
            f_w = np.ones(Nf)
        elif lattice == "sq":
            assert Nf == int(Nf ** (1 / s) + 0.5) ** s
            ns = int(Nf ** (1 / s))
            X, Y = np.meshgrid(np.arange(0, 1, 1 / ns) + np.random.uniform(0, 1 / ns), np.arange(0, 1, 1 / ns) + np.random.uniform(0, 1 / ns))
            X_0 = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
            f_w = np.ones(Nf)
        elif lattice == "sob":
            if GLT.sampler is None:
                GLT.sampler = Sobol(d=s, scramble=True)
            try:
                X_0 = GLT.sampler.random(n=Nf)
            except:
                GLT.sampler.reset()
                X_0 = GLT.sampler.random(n=Nf)
            f_w = np.ones(Nf)
        elif lattice == "glt":
            X_0 = GLT.get_lattice(Nf, time_reflected, space_reflected)
            f_w = np.ones(Nf)
        elif lattice == "gau":
            assert Nf**0.5 == int(Nf**0.5)
            ns = int(Nf**0.5)
            if ns not in GLT.roots_legendre:
                import scipy.special

                x, w = scipy.special.roots_legendre(ns)
                x = (x + 1) / 2  # [0,1]
                X, Y = np.meshgrid(x, x)
                X_0 = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
                f_w = w[:, None] * w[None, :]
                f_w = f_w.flatten() / f_w.mean()
                GLT.roots_legendre[ns] = (X_0, f_w)
            X_0, f_w = GLT.roots_legendre[ns]
        else:
            raise NotImplementedError("invalid lattice:", lattice)
        X_f = lb + (ub - lb) * X_0
        return X_f, f_w
