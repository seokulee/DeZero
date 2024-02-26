import numpy as np

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v

class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-7):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = np.zeros_like(param.data)

        h = self.hs[h_key]
        h += param.grad.data * param.grad.data
        param.data -= self.lr * param.grad.data / (np.sqrt(h) + self.eps)

class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6):
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def update_one(self, param):
        key = id(param)
        if key not in self.msg:
            self.msg[key] = np.zeros_like(param.data)
            self.msdx[key] = np.zeros_like(param.data)

        msg, msdx = self.msg[key], self.msdx[key]
        g = param.grad.data
        msg *= self.rho
        msg += (1 - self.rho) * g * g

        dx = np.sqrt((msdx + self.eps) / (msg + self.eps)) * g
        msdx *= self.rho
        msdx += (1 - self.rho) * dx * dx

        param.data -= dx

class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update_one(self, param):
        key = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2 = self.beta1, self.beta2
        g = param.grad.data

        m += (1 - beta1) * (g - m)
        v += (1 - beta2) * (g * g - v)
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)

        param.data -= self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)
