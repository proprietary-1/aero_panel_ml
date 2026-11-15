"""A small fully-connected neural network written from scratch in numpy.

Tanh hidden layers, linear output, Adam optimiser, input/output standardisation,
and a vector-Jacobian product with respect to the *inputs*, so the trained
surrogate can drive gradient-based design optimisation.
"""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.normal(0, np.sqrt(1.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.x_mu = self.x_sd = self.y_mu = self.y_sd = None

    # ---------- core maths (standardised space) ----------
    def _forward(self, z):
        acts = [z]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = z @ W + b
            if i < len(self.W) - 1:
                z = np.tanh(z)
            acts.append(z)
        return acts

    def _backward(self, acts, dout):
        """Backprop dout through the net. Returns (dW, db, d_input)."""
        dW, db = [None] * len(self.W), [None] * len(self.W)
        g = dout
        for i in reversed(range(len(self.W))):
            dW[i] = acts[i].T @ g
            db[i] = g.sum(axis=0)
            g = g @ self.W[i].T
            if i > 0:
                g = g * (1 - acts[i] ** 2)  # derivative of tanh
        return dW, db, g

    # ---------- training ----------
    def fit(self, X, Y, X_val, Y_val, epochs=2000, lr=3e-3, batch=128,
            weight_decay=1e-6, patience=300, seed=0, verbose=True):
        self.x_mu, self.x_sd = X.mean(0), X.std(0)
        self.y_mu, self.y_sd = Y.mean(0), Y.std(0)
        Xs, Ys = (X - self.x_mu) / self.x_sd, (Y - self.y_mu) / self.y_sd
        Xv, Yv = (X_val - self.x_mu) / self.x_sd, (Y_val - self.y_mu) / self.y_sd

        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        best, best_state, since_best, history = np.inf, None, 0, []

        for ep in range(epochs):
            lr_ep = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))  # cosine decay
            idx = rng.permutation(len(Xs))
            for s in range(0, len(idx), batch):
                bi = idx[s:s + batch]
                acts = self._forward(Xs[bi])
                dout = 2 * (acts[-1] - Ys[bi]) / len(bi)
                dW, db, _ = self._backward(acts, dout)
                grads = [g + weight_decay * p for g, p in zip(dW + db, params)]
                step += 1
                for k, (p, g) in enumerate(zip(params, grads)):
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    p -= lr_ep * (m[k] / (1 - b1**step)) / (np.sqrt(v[k] / (1 - b2**step)) + eps)
            tr = np.mean((self._forward(Xs)[-1] - Ys) ** 2)
            va = np.mean((self._forward(Xv)[-1] - Yv) ** 2)
            history.append((tr, va))
            if va < best:
                best, since_best = va, 0
                best_state = ([w.copy() for w in self.W], [c.copy() for c in self.b])
            else:
                since_best += 1
            if verbose and ep % 200 == 0:
                print(f"  epoch {ep:5d}  train {tr:.2e}  val {va:.2e}")
            if since_best > patience:
                break
        # restore best weights in place (params list aliases these arrays)
        for w, wb in zip(self.W, best_state[0]):
            w[...] = wb
        for c, cb in zip(self.b, best_state[1]):
            c[...] = cb
        return np.array(history)

    # ---------- inference ----------
    def predict(self, X):
        Xs = (np.atleast_2d(X) - self.x_mu) / self.x_sd
        return self._forward(Xs)[-1] * self.y_sd + self.y_mu

    def vjp_input(self, X, dY):
        """Gradient of sum(dY * f(X)) with respect to X, in physical units."""
        Xs = (np.atleast_2d(X) - self.x_mu) / self.x_sd
        acts = self._forward(Xs)
        _, _, g = self._backward(acts, np.atleast_2d(dY) * self.y_sd)
        return g / self.x_sd

    # ---------- persistence ----------
    def save(self, path):
        d = {f"W{i}": w for i, w in enumerate(self.W)}
        d.update({f"b{i}": c for i, c in enumerate(self.b)})
        np.savez(path, sizes=np.array(self.sizes), x_mu=self.x_mu, x_sd=self.x_sd,
                 y_mu=self.y_mu, y_sd=self.y_sd, **d)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(d["sizes"].tolist())
        net.W = [d[f"W{i}"] for i in range(len(net.W))]
        net.b = [d[f"b{i}"] for i in range(len(net.b))]
        net.x_mu, net.x_sd, net.y_mu, net.y_sd = d["x_mu"], d["x_sd"], d["y_mu"], d["y_sd"]
        return net
