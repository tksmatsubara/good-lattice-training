import sys

sys.path.insert(0, "../../Utilities/")
import time
import os
import argparse
import tensorflow as tf
import numpy as np
from pyDOE import lhs

from glt2 import GLTsD


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--noretry", dest="noretry", action="store_true", help="ignore finished trials.")
    # experimental setting
    parser.add_argument("--dataset", default="Poisson", type=str, help="dataset")
    parser.add_argument("--K", default=2, type=int, help="freq of Poisson")
    parser.add_argument("--s", default=2, type=int, help="number of dimensions")
    parser.add_argument("--seed", default=0, type=int, help="random seeds")
    parser.add_argument("--nf", default=20000, type=int, help="number of collocation points")
    parser.add_argument("--boundary", default="forcing", type=str, help="enforce the boundary condition", choices=["forcing"])
    # uni for uniformly random sampling
    # sq for uniformly spaced sampling
    # lhs for Latin hypercube sampling
    # glt for good lattice training
    parser.add_argument("--lattice", default="lhs", type=str, help="method to determine collocation points")
    # fixed: use the same points
    # random: get new points at every step
    parser.add_argument("--sample", default="fixed", type=str, help="update collocation points or not")
    # default: PINNs default. Adam + L-BFGS-B
    # cosine: Adam + cosine annealing
    # adam: Adam
    parser.add_argument("--strategy", default="default", type=str, help="optimizer to use", choices=["default", "cosine", "adam"])
    # 50,000 for default strategy
    # 200,000 for default cosine anealing
    parser.add_argument("--n_itr", default=None, type=int, help="number of iterations for Adam")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate for Adam")
    # logging
    parser.add_argument("--log_freq", default=1000, type=int, help="number of gradient steps between prints")
    parser.add_argument("--postfix", default="", type=str, help="postfix string for log file")
    # network
    parser.add_argument("--n_hidden", default=100, type=int, help="number of hidden units")
    parser.add_argument("--n_layers", default=4, type=int, help="number of hidden layers")
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    if args.n_itr is None:
        if args.strategy == "default":
            args.n_itr = 50000
        elif args.strategy in ["cosine", "adam"]:
            args.n_itr = 200000
        else:
            raise NotImplementedError
    return args


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, args, dataset, Nf, lattice, boundary):
        self.dataset = dataset
        self.lb = self.dataset.lb
        self.ub = self.dataset.ub
        self.s = self.dataset.s
        self.boundary = boundary
        self.lr_default = args.lr

        self.lattice = lattice
        self.K = args.K
        self.Nf = Nf
        X_f, f_w = GLTsD.get_evaluation_points(self.lattice, self.s, self.lb, self.ub, self.Nf)

        # training data
        self.X_f = X_f[:]
        self.f_w = f_w[:, np.newaxis]

        # test data
        self.X_star = self.dataset.X_star
        self.u_star = self.dataset.u_star

        # Initialize NNs
        self.layers = [self.s] + [args.n_hidden] * args.n_layers + [dataset.state_dim]
        self.weights, self.biases = self.initialize_NN(self.layers)

        # tf Placeholders
        self.f_w_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.lr_tf = tf.placeholder(tf.float64, shape=())
        self.X_f_tf = tf.placeholder(tf.float64, shape=[None, self.X_f.shape[1]])

        # tf Graphs
        self.u_pred = self.net_u(self.X_f_tf)
        self.f_u_pred = self.net_f_u(self.X_f_tf)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.f_u_pred) * self.f_w_tf)  # loss from equation

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method="L-BFGS-B", options={"maxiter": 50000, "maxfun": 50000, "maxcor": 50, "maxls": 50, "ftol": 1.0 * np.finfo(float).eps})
        # np.finfo(float).eps=2.220446049250313e-16

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.lr_tf)
        # self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        # if a position is exactly at the edge, the gradient vanishes.
        grads_vars = self.optimizer_Adam.compute_gradients(self.loss)
        grads_vars = [(tf.where(tf.is_nan(vs[0]), tf.zeros_like(vs[0]), vs[0]), vs[1]) for vs in grads_vars]
        self.train_op_Adam = self.optimizer_Adam.apply_gradients(grads_vars)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * X - 1.0  # assumed already [0,1]
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, X):
        X_normed = (X - self.lb) / (self.ub - self.lb)
        u_raw = self.neural_net(X_normed, self.weights, self.biases)
        u = u_raw * tf.reduce_prod(X_normed * (1 - X_normed), axis=-1, keepdims=True)
        return u

    def net_f_u(self, X):
        xs = [X[..., i : i + 1] for i in range(self.s)]
        u = self.net_u(tf.concat(xs, axis=-1))

        f_u = 0
        for i in range(self.s):
            u_x = tf.gradients(u, xs[i])[0]
            u_xx = tf.gradients(u_x, xs[i])[0]
            f_u = f_u + u_xx

        f_u = f_u + tf.reduce_prod(tf.sin(self.K * np.pi * X), axis=-1, keepdims=True) * (self.s * self.K**2 * np.pi**2)
        return f_u

    def train(self, n_itr, log_freq=10, sample="fixed", strategy="default"):
        tf_dict = {
            self.X_f_tf: self.X_f,
            self.lr_tf: self.lr_default,
            self.f_w_tf: self.f_w,
        }

        self.start_time = time.time()
        for itr in range(n_itr + 1):
            if itr % log_freq == 0:
                elapsed = time.time() - self.start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_value_test = self.get_test_loss()
                print("Adam It: %d, Loss: %.3e, Test Loss: %.3e, Time: %.2f" % (itr, loss_value, loss_value_test[-1], elapsed))
                self.print_log(itr, loss_value, *loss_value_test, elapsed)
                self.start_time = time.time()
            if itr == n_itr:
                break

            if sample == "random":
                X_f, f_w = GLTsD.get_evaluation_points(self.lattice, self.s, self.lb, self.ub, self.Nf)
                tf_dict[self.X_f_tf] = self.X_f = X_f[:, :]  # evaluation points
                tf_dict[self.f_w_tf] = self.f_w = f_w[:, np.newaxis]

            if strategy == "cosine":
                learning_rate_scale = np.cos(itr / n_itr * np.pi) / 2 + 0.5
                tf_dict[self.lr_tf] = self.lr_default * learning_rate_scale

            self.sess.run(self.train_op_Adam, tf_dict)

        if strategy == "default":
            self.start_time = time.time()
            self.itr = n_itr

            def callback(loss_value):
                if self.itr % log_freq == 0:
                    elapsed = time.time() - self.start_time
                    print("L-BFGS-B It: %d, Loss: %.3e, Time: %.2f" % (self.itr, loss_value, elapsed))
                    self.start_time = time.time()
                self.itr += 1

            self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=callback)
            loss_value_test = self.get_test_loss()
            print("Finalize It: %d, Loss: %.3e, Test Loss: %.3e, Time: %.2f" % (self.itr, 0.0, loss_value_test[-1], 0.0))
            self.print_log(self.itr, 0.0, *loss_value_test, 0.0)

            del self.start_time
            del self.itr
        elif strategy == "cosine":
            loss_value_test = self.get_test_loss()
            print("Finalize It: %d, Loss: %.3e, Test Loss: %.3e, Time: %.2f" % (n_itr, 0.0, loss_value_test[-1], 0.0))
            self.print_log(n_itr, 0.0, *loss_value_test, 0.0)

    def predict(self, X_star):
        tf_dict = {self.X_f_tf: X_star}

        u_star = self.sess.run(self.u_pred, tf_dict)

        return u_star

    def initialize_test_data(self, X_star, u_star):
        self.X_star = X_star
        self.u_star = u_star

    def get_test_loss(self):
        u_pred = model.predict(self.X_star)

        error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
        return (error_u,)

    def initialize_logger(self, log_file_name, *args):
        self.log_file = open(log_file_name, "w")
        print(f"# {args[0]}", *(args[1:]), sep="\t", file=self.log_file)

    def print_log(self, *args):
        print(*args, sep="\t", file=self.log_file)
        self.log_file.flush()


class PoissonDataset:
    def __init__(self, s, K):
        self.s = s
        self.lb = np.array([0.0] * s)  # lower bound of (x,y,z,w,...)
        self.ub = np.array([1.0] * s)  # upper bound of (x,y,z,w,...)
        self.state_dim = 1

        n_div = {
            2: 1000,  # (1,000-1)^2=1,000,000
            4: 50,  # (50-1)^4=5,764,801
            5: 23,  # (23-1)^5=5,153,632
            6: 14,  # (14-1)^6=4,826,809
            7: 10,  # (10-1)^7=4,782,969
            8: 8,  # (8-1)^8=5,764,801
        }

        xs = [np.linspace(self.lb[i], self.ub[i], n_div[s] + 1)[1:-1] for i in range(s)]
        Xs = np.meshgrid(*xs)
        Exact_u = np.prod([np.sin(K * np.pi * Xs[i]) for i in range(args.s)], axis=0)
        self.X_star = np.stack([Xs[i].flatten() for i in range(args.s)], axis=-1)  # (n_div-1)^s x s
        self.u_star = Exact_u.flatten()[:, None]  # (n_div-1)^s x 1


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists("results"):
        os.makedirs("results")
    args.log_file_name = f"results/{args.dataset}-s{args.s}K{args.K}{args.postfix}-{args.lattice}-{args.boundary}-{args.sample}-{args.strategy}-{args.nf}-seed{args.seed}.txt"
    if os.path.exists(args.log_file_name):
        if args.noretry:
            print(args.log_file_name)
            print("====== already done:", " ".join(sys.argv), flush=True)
            exit()

    np.random.seed(1234 + args.seed)
    tf.set_random_seed(1234 + args.seed)

    ###########################
    dataset = PoissonDataset(args.s, args.K)

    model = PhysicsInformedNN(args, dataset, args.nf, args.lattice, args.boundary)
    # model = PhysicsInformedNN(layers, lb, ub, Nf, args.lattice, args.K)
    model.initialize_logger(args.log_file_name, "itr", "train", "u")

    start_time = time.time()
    model.train(args.n_itr, log_freq=args.log_freq, sample=args.sample, strategy=args.strategy)
    elapsed = time.time() - start_time
    print("Training time: %.4f" % (elapsed))

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    args.model_file_name = args.log_file_name.replace(".txt", ".ckpt")
    saver = tf.train.Saver()
    saver.save(model.sess, args.model_file_name)
    args.data_file_name = args.log_file_name.replace(".txt", ".npy")
    np.save(args.data_file_name, model.predict(dataset.X_star))
