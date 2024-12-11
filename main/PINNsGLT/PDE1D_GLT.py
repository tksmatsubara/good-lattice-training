import sys
import time
import os
import argparse
import tensorflow as tf
import numpy as np
import scipy.io

from glt import GLT

our_dtype = tf.float32


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--noretry", dest="noretry", action="store_true", help="ignore finished trials.")
    # experimental setting
    parser.add_argument("--dataset", default="NLS", type=str, help="dataset", choices=["NLS", "AC", "KdV"])
    parser.add_argument("--seed", default=0, type=int, help="random seeds")
    parser.add_argument("--nf", default=20000, type=int, help="number of collocation points")
    # forcing: ensuring the initial and boundary conditions; for GLT, the time coordinate is folded.
    # learning: learning the initial and boundary conditions; for GLT, the time/space coordinates are folded.
    # forcing0: ensuring the boundary condition, but not initial condition; for GLT, the time coordinate is folded.
    # learning0: not ensuring any conditions; for GLT, no coordinate is folded.
    parser.add_argument("--boundary", default="forcing", type=str, help="enforce the boundary condition", choices=["forcing", "learning", "learning0", "forcing0"])
    # uni for uniformly random sampling
    # sq for uniformly spaced sampling
    # lhs for Latin hypercube sampling
    # glt for good lattice training
    # sob for Sobol sequence
    parser.add_argument("--lattice", default="lhs", type=str, help="method to determine collocation points")
    # fixed: use the same points
    # random: get new points at every step
    parser.add_argument("--sample", default="fixed", type=str, help="update collocation points or not")
    # default: PINNs default. Adam + L-BFGS-B
    # cosine: Adam + cosine annealing
    parser.add_argument("--strategy", default="default", type=str, help="optimizer to use")
    # 50,000 for default strategy
    # 200,000 for default cosine anealing
    parser.add_argument("--n_itr", default=None, type=int, help="number of iterations for Adam")
    # logging
    parser.add_argument("--log_freq", default=100, type=int, help="number of gradient steps between prints")
    parser.add_argument("--postfix", default="", type=str, help="postfix string for log file")
    # network
    parser.add_argument("--n_hidden", default=100, type=int, help="number of hidden units")
    parser.add_argument("--n_layers", default=4, type=int, help="number of hidden layers")
    # additional experiments
    parser.add_argument("--identification", dest="identification", action="store_true", help="system identification.")
    parser.add_argument("--n_obs", default=100, type=int, help="number of observations for system identification")
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    if args.n_itr is None:
        if args.strategy == "default":
            args.n_itr = 50000
        elif args.strategy == "cosine":
            args.n_itr = 200000
        else:
            raise NotImplementedError
    return args


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, args, dataset, Nf, lattice, boundary, data_for_system_idenfitication=None):
        self.dataset = dataset
        self.lb = self.dataset.lb
        self.ub = self.dataset.ub
        self.boundary = boundary
        self.is_system_identification = data_for_system_idenfitication is not None

        self.initial_condition = self.dataset.get_initial_condition()

        # evaluation points in domain
        self.lattice = lattice
        self.Nf = Nf
        X_f, f_w = GLT.get_evaluation_points(self.lattice, self.lb, self.ub, self.Nf, space_reflected=self.boundary == "learning", time_reflected=not self.boundary == "learning0")

        # training data
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.f_w = f_w[:, np.newaxis]

        # test data
        self.X_star = self.dataset.X_star
        self.u_star = self.dataset.u_star

        # Initialize NNs
        self.layers = [(2 if self.boundary.startswith("forcing") else 1) + 1] + [args.n_hidden] * args.n_layers + [dataset.state_dim]
        self.weights, self.biases = self.initialize_NN(self.layers)

        # tf Placeholders
        self.f_w_tf = tf.placeholder(our_dtype, shape=[None, self.x_f.shape[1]])
        self.lr_tf = tf.placeholder(our_dtype, shape=())

        self.x_f_tf = tf.placeholder(our_dtype, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(our_dtype, shape=[None, self.t_f.shape[1]])

        # tf Graphs
        self.u_pred = self.net_u(self.x_f_tf, self.t_f_tf)
        if self.is_system_identification:
            self.l1 = tf.Variable(tf.zeros(1, dtype=our_dtype), dtype=our_dtype)
            self.l2 = tf.Variable(tf.zeros(1, dtype=our_dtype), dtype=our_dtype)
            self.f_u_pred = self.net_f_u(self.x_f_tf, self.t_f_tf, params=(self.l1, self.l2))
        else:
            self.f_u_pred = self.net_f_u(self.x_f_tf, self.t_f_tf)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.f_u_pred[:, 0:1]) * self.f_w_tf)  # loss from equation
        if self.u_star.shape[1] == 2:
            self.loss = self.loss + tf.reduce_mean(tf.square(self.f_u_pred[:, 1:2]) * self.f_w_tf)  # loss from equation

        if self.boundary in ["learning", "forcing0", "learning0"]:
            self.learning_initial_condition()
        if self.boundary in ["learning", "learning0"]:
            self.learning_boundary_condition()

        # training data for system identification
        if self.is_system_identification:
            assert data_for_system_idenfitication is not None
            self.x_o_tf = tf.cast(data_for_system_idenfitication[0], our_dtype)
            self.t_o_tf = tf.cast(data_for_system_idenfitication[1], our_dtype)
            self.u_o_tf = tf.cast(data_for_system_idenfitication[2], our_dtype)
            self.u_o_pred = self.net_u(self.x_o_tf, self.t_o_tf)
            self.loss = self.loss + tf.reduce_mean(tf.square(self.u_o_tf - self.u_o_pred))

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method="L-BFGS-B", options={"maxiter": 50000, "maxfun": 50000, "maxcor": 50, "maxls": 50, "ftol": 1.0 * np.finfo(float).eps})
        # np.finfo(float).eps=2.220446049250313e-16

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.lr_tf)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def learning_initial_condition(self):
        x0 = self.dataset.x0
        u0 = self.dataset.u0
        X0 = np.concatenate((x0, 0 * x0), 1)  # add t=0 to initial condition

        # initial condition
        self.x0 = tf.cast(X0[:, 0:1], our_dtype)
        self.t0 = tf.cast(X0[:, 1:2], our_dtype)
        self.u0 = tf.cast(u0, our_dtype)

        self.u0_pred = self.net_u(self.x0, self.t0)
        self.loss = self.loss + tf.reduce_mean(tf.square(self.u0[:, 0:1] - self.u0_pred[:, 0:1]))

        # for NLS
        if self.u0.shape[1] == 2:
            self.loss = self.loss + tf.reduce_mean(tf.square(self.u0[:, 1:2] - self.u0_pred[:, 1:2]))

    def learning_boundary_condition(self):
        lb = self.dataset.lb
        ub = self.dataset.ub
        tb = self.dataset.tb

        X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # add x at lower boundary to boundary condition
        X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # add x at upper boundary to boundary condition

        self.x_lb = tf.cast(X_lb[:, 0:1], our_dtype)
        self.t_lb = tf.cast(X_lb[:, 1:2], our_dtype)

        self.x_ub = tf.cast(X_ub[:, 0:1], our_dtype)
        self.t_ub = tf.cast(X_ub[:, 1:2], our_dtype)

        # boundary condition
        u_lb_pred = self.net_u(self.x_lb, self.t_lb)
        u_ub_pred = self.net_u(self.x_ub, self.t_ub)
        self.loss = self.loss + tf.reduce_mean(tf.square(u_lb_pred[:, 0:1] - u_ub_pred[:, 0:1]))

        # boundary condition, gradient
        u_x_lb_pred = tf.gradients(u_lb_pred[:, 0:1], self.x_lb)[0]
        u_x_ub_pred = tf.gradients(u_ub_pred[:, 0:1], self.x_ub)[0]
        self.loss = self.loss + tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred))

        # for NLS
        if self.u0.shape[1] == 2:
            self.loss = self.loss + tf.reduce_mean(tf.square(u_lb_pred[:, 1:2] - u_ub_pred[:, 1:2]))
            v_x_lb_pred = tf.gradients(u_lb_pred[:, 1:2], self.x_lb)[0]
            v_x_ub_pred = tf.gradients(u_ub_pred[:, 1:2], self.x_ub)[0]
            self.loss = self.loss + tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred))

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=our_dtype), dtype=our_dtype)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=our_dtype), dtype=our_dtype)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = X  # already normalized
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        x_ub, t_ub = self.ub
        x_lb, t_lb = self.lb
        x_normed = (x - x_lb) / (x_ub - x_lb)  # [0,1]
        t_normed = (t - t_lb) / (t_ub - t_lb)  # [0,1]
        if self.boundary.startswith("forcing"):  # forcing periodic boundary condition
            x1 = tf.cos(2 * np.pi * x_normed)
            x2 = tf.sin(2 * np.pi * x_normed)
            X = tf.concat([x1, x2, 2 * t_normed - 1], 1)
        else:
            X = tf.concat([2 * x_normed - 1, 2 * t_normed - 1], 1)

        u = self.neural_net(X, self.weights, self.biases)
        u0 = self.initial_condition(x)

        if self.boundary.startswith("forcing"):  # forcing initial condition
            if self.boundary == "forcing0":
                pass
            else:
                tau = tf.cast(1.0, our_dtype)
                u = tf.exp(-t_normed / tf.exp(tau)) * u0 + (1 - tf.exp(-t_normed / tf.exp(tau))) * u
        return u

    def net_f_u(self, x, t, params=None):
        u = self.net_u(x, t)
        return self.dataset.get_fu_from_u(x, t, u, params=params)

    def train(self, n_itr, log_freq=10, sample="fixed", strategy="default", observation=None):
        tf_dict = {
            self.x_f_tf: self.x_f,
            self.t_f_tf: self.t_f,
            self.lr_tf: 0.001,
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
                X_f, f_w = GLT.get_evaluation_points(self.lattice, self.lb, self.ub, self.Nf, space_reflected=self.boundary == "learning", time_reflected=not self.boundary == "learning0")
                tf_dict[self.x_f_tf] = self.x_f = X_f[:, 0:1]  # evaluation points
                tf_dict[self.t_f_tf] = self.t_f = X_f[:, 1:2]  # evaluation points
                tf_dict[self.f_w_tf] = self.f_w = f_w[:, np.newaxis]

            if strategy == "cosine":
                learning_rate_scale = np.cos(itr / n_itr * np.pi) / 2 + 0.5
                tf_dict[self.lr_tf] = 0.001 * learning_rate_scale

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
        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}
        u_pred = self.sess.run(self.u_pred, tf_dict)
        return u_pred

    def get_test_loss(self):
        u_pred = model.predict(self.X_star)
        if u_pred.shape[1] == 1:
            error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
            return (error_u,)
        else:
            r_star = self.u_star
            r_pred = u_pred
            u_star, v_star = self.u_star[:, 0:1], self.u_star[:, 1:2]
            u_pred, v_pred = u_pred[:, 0:1], u_pred[:, 1:2]
            h_star = np.sqrt(u_star**2 + v_star**2)
            h_pred = np.sqrt(u_pred**2 + v_pred**2)

            error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
            error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
            error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
            error_r = np.linalg.norm(r_star - r_pred, 2) / np.linalg.norm(r_star, 2)
            return error_u, error_v, error_h, error_r

    def initialize_logger(self, log_file_name, *args):
        self.log_file = open(log_file_name, "w")
        print(f"# {args[0]}", *(args[1:]), sep="\t", file=self.log_file)

    def print_log(self, *args):
        if self.is_system_identification:
            print(*args, *[v[0] for v in model.sess.run([self.l1, self.l2])], sep="\t", file=self.log_file)
        else:
            print(*args, sep="\t", file=self.log_file)
        self.log_file.flush()


class Dataset:
    def __init__(self, name, N0=50, Nb=50):
        self.name = name
        data = scipy.io.loadmat(f"../Data/{name}.mat")
        if self.name == "NLS":
            # x: 1x256 points, np.arange(-5,5,10/256)
            # t: 1x201 points, around np.linspace(0,np.pi/2,201)
            self.lb = np.array([-5.0, 0.0])  # lower bound of (x,t)
            self.ub = np.array([5.0, np.pi / 2])  # upper bound of (x,t)
            self.state_dim = 2
        elif self.name in ["KdV", "AC"]:
            # x: 1x512 points, np.arange(-1,1,513)[:-1]
            # t: 1x201 points, around np.linspace(0,1,201)
            self.lb = np.array([-1.0, 0.0])  # lower bound of (x,t)
            self.ub = np.array([1.0, 1.0])  # upper bound of (x,t)
            self.state_dim = 1
        else:
            raise NotImplementedError(name)

        x = data["x"].flatten()[:, None]
        t = data["tt"].flatten()[:, None]
        data_u = data["uu"]  # (x,t)-space
        X, T = np.meshgrid(x, t)
        self.X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # X x T x 2
        if self.name == "NLS":
            data_u = np.stack([np.real(data_u), np.imag(data_u)], axis=-1)
            self.u_star = data_u.transpose(1, 0, 2).reshape(-1, 2)
        else:
            data_u = data_u[..., None]
            self.u_star = data_u.T.flatten()[:, None]  # X x T x 1

        idx_x = np.random.choice(x.shape[0], N0, replace=False)  # N0 points from x
        self.x0 = x[idx_x, :]
        self.u0 = data_u[idx_x, 0]

        idx_t = np.random.choice(t.shape[0], Nb, replace=False)  # N_b points from t
        self.tb = t[idx_t, :]  # boundary condition

    def get_initial_condition(self):
        if self.name == "NLS":
            return lambda x: tf.concat([2 / tf.cosh(x), x * 0], axis=-1)
        elif self.name == "KdV":
            return lambda x: tf.cos(np.pi * x)
        elif self.name == "AC":
            return lambda x: x**2 * tf.cos(np.pi * x)
        else:
            raise NotImplementedError(self.name)

    def get_fu_from_u(self, x, t, u, params=None):
        if self.name == "NLS":
            u, v = u[..., 0:1], u[..., 1:2]

            u_t = tf.gradients(u, t)[0]
            u_x = tf.gradients(u, x)[0]
            u_xx = tf.gradients(u_x, x)[0]

            v_t = tf.gradients(v, t)[0]
            v_x = tf.gradients(v, x)[0]
            v_xx = tf.gradients(v_x, x)[0]

            f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
            f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u
            return tf.concat([f_u, f_v], axis=-1)
        elif self.name == "KdV":
            u_t = tf.gradients(u, t)[0]
            u_x = tf.gradients(u, x)[0]
            u_xx = tf.gradients(u_x, x)[0]
            u_xxx = tf.gradients(u_xx, x)[0]
            if params is None:
                l1 = 1.0
                l2 = 0.0025
            else:
                l1, l2 = params
            f_u = u_t + l1 * u * u_x + l2 * u_xxx
            return f_u
        elif self.name == "AC":
            u_t = tf.gradients(u, t)[0]
            u_x = tf.gradients(u, x)[0]
            u_xx = tf.gradients(u_x, x)[0]
            if params is None:
                l1 = 0.0001
                l2 = 5.0
            else:
                l1, l2 = params
            # f_u = u_t - 0.0001 * u_xx + 5 * u**3 - 5 * u
            # f_u = u_t - (0.0001 * u_xx - 5 * u**3 + 5 * u)
            # f_u = u_t - (0.0001 * u_xx - 5* (u**3 - u))
            # f_u = u_t - (l1 * u_xx - l2 * (u**3 - u))
            f_u = u_t - (l1 * u_xx - l2 * (u**3 - u))
            return f_u
        else:
            raise NotImplementedError(self.name)


if __name__ == "__main__":
    args = get_args()
    if args.lattice == "glt":
        assert args.nf in GLT.table.keys()

    if args.identification:
        if not os.path.exists(f"results_identification"):
            os.makedirs(f"results_identification")
        args.log_file_name = f"results_identification/{args.dataset}{args.postfix}-{args.n_obs}obs-{args.lattice}-{args.boundary}-{args.sample}-{args.strategy}-{args.nf}-seed{args.seed}.txt"
    else:
        if not os.path.exists("results"):
            os.makedirs("results")
        args.log_file_name = f"results/{args.dataset}{args.postfix}-{args.lattice}-{args.boundary}-{args.sample}-{args.strategy}-{args.nf}-seed{args.seed}.txt"
    if os.path.exists(args.log_file_name) and args.noretry:
        print(args.log_file_name)
        print("====== already done:", " ".join(sys.argv), flush=True)
        exit()
    else:
        with open(args.log_file_name, "w"):
            pass

    np.random.seed(1234 + args.seed)
    tf.set_random_seed(1234 + args.seed)

    noise = 0.0

    ###########################
    dataset = Dataset(args.dataset)

    if args.identification:
        rng_state = np.random.get_state()
        np.random.seed(1001)
        index_obs = np.random.choice(dataset.u_star.shape[0], args.n_obs, replace=False)
        x_obs = dataset.X_star[:, :1][index_obs]
        t_obs = dataset.X_star[:, 1:][index_obs]
        u_obs = dataset.u_star[index_obs]
        data_for_system_idenfitication = [x_obs, t_obs, u_obs]
        np.random.set_state(rng_state)
    else:
        data_for_system_idenfitication = None
    model = PhysicsInformedNN(args, dataset, args.nf, args.lattice, args.boundary, data_for_system_idenfitication=data_for_system_idenfitication)
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
    if model.is_system_identification:
        args.params_file_name = args.log_file_name.replace(".txt", "_params.txt")
        np.savetxt(args.log_file_name.replace(".txt", "_params.txt"), model.sess.run([model.l1, model.l2]))
