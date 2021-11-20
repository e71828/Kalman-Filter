import numpy as np
rng = np.random.default_rng()


class KF:
    # Kalman Filter Recursive Algorithm
    # K_k = P^prime_k * H_k^T * [(H_k * P_k^prime * H_k^T + R_k)^-1]

    def __init__(self, x0: list or np.ndarray,
                 P0: list or np.ndarray,
                 H: list or np.ndarray,
                 Q: list or np.ndarray,
                 R: list or np.ndarray,
                 PHI: list or np.ndarray,
                 GAMMA: list or np.ndarray,
                 UPSILON: list or np.ndarray,
                 ) -> None:
        # assuming two important args at t0
        # x^prime(t0) = x0
        # P0^prime = E{x^tilde(t0) (x^tilde(t0))^T}

        # for one dimension
        if isinstance(x0, list):
            self.x_prime_k = np.array(x0)  # initial state
            self.n = np.size(self.x_prime_k, 0)
            self.P_prime_k = np.array(P0)
            self.H_k = np.array(H)  # measure matrix: m x n
            self.m = np.size(self.H_k, 0)
            self.PHI_k = np.array(PHI)  # process matrix
            self.GAMMA_k = np.array(GAMMA)
            self.UPSILON_k = np.array(UPSILON)  # Square root of precess noise Q_k, for w_k ~ N(0, 1)
            self.Q_k = np.array(Q)  # process noise
        # for two dimension
        else:
            self.x_prime_k = x0  # initial state
            self.n = np.size(self.x_prime_k, 0)
            self.P_prime_k = P0
            self.H_k = H  # measure matrix: m x n
            self.m = np.size(self.H_k, 0)
            self.PHI_k = PHI  # process matrix
            self.GAMMA_k = GAMMA
            self.UPSILON_k = UPSILON  # Square root of precess noise Q_k, for w_k ~ N(0, 1)
            self.Q_k = Q  # process noise

        if self.m == 1:
            self.R_k = np.array(R)  # measurement noise
        else:
            self.R_k = R

        # record
        self.P_prime, self.x_prime, self.K, self.x_hat, self.P = [], [], [], [], []
        self.y, self.mean = [], []

        # intermediate variable
        self.K_k, self.P_k = np.array([]), np.array([])
        self.y_k, self.u_k, self.x_hat_k = np.array([]), np.array([]), np.array([])

    def gain(self) -> None:
        # K_k = P^prime_k * H_k^T * [(H_k * P_k^prime * H_k^T + R_k)^-1]
        # Stationary point : [H_k * P^prime_k]^T = K_k * (H_k * P_k^prime * H_k^T + R_k)

        if self.m == 1 and self.n == 1:
            S_k = self.H_k * self.P_prime_k * self.H_k + self.R_k
            self.K_k = self.P_prime_k * self.H_k / S_k
        elif self.m != 1 and self.n == 1:
            S_k = self.H_k * self.P_prime_k @ self.H_k.reshape(1, self.m) + self.R_k
            self.K_k = self.P_prime_k * self.H_k.reshape(1, self.m) @ np.linalg.inv(S_k)
        elif self.m == 1 and self.n != 1:
            S_k = self.H_k @ self.P_prime_k @ self.H_k.reshape(self.n, 1) + self.R_k
            self.K_k = self.P_prime_k @ self.H_k.reshape(self.n, 1) / S_k
        else:
            S_k = self.H_k @ self.P_prime_k @ self.H_k.T + self.R_k
            self.K_k = self.P_prime_k @ self.H_k.T @ np.linalg.inv(S_k)

    def update_estimation(self, y_k) -> None:
        # i_k  = y_k - H_k * x_prime_k
        # x_hat_k = x_prime_k + K_k * i_k
        self.y_k = np.array(y_k).reshape(self.m)

        i_k = self.y_k - self.H_k.dot(self.x_prime_k.reshape(self.n))
        self.x_hat_k = self.x_prime_k + self.K_k.dot(i_k)

    def update_covariance(self) -> None:
        # P_k = (I - K_k * H_k) * P_prime_k
        self.P_k = (np.eye(self.n) - self.K_k @ self.H_k) @ self.P_prime_k

    def propagation(self, u_k) -> None:
        # project into k+1
        # x_prime_(k+1) = PHI_k * x_k + GAMMA_k * u_k +  UPSILON_k * w_k
        # w_k ~ Normal(0, Q_k)
        # P_prime_(K+1) = PHI_k * P_k * PHI_k^T + UPSILON_k * Q_k * UPSILON_k^T
        if u_k:
            self.u_k = np.array(u_k)
        else:
            self.u_k = np.zeros(self.n)

        x_k_prime = self.PHI_k.dot(self.x_hat_k) + self.GAMMA_k.dot(self.u_k)
        if not isinstance(x_k_prime, np.ndarray):
            x_k_prime = np.array([x_k_prime])
        if self.n == 1:
            P_k_prime = self.PHI_k * self.P_k * self.PHI_k.T + self.UPSILON_k * self.Q_k * self.UPSILON_k.T
        elif np.allclose(self.UPSILON_k, np.eye(self.n)):
            P_k_prime = self.PHI_k @ self.P_k @ self.PHI_k.T + self.Q_k
        else:
            P_k_prime = self.PHI_k @ self.P_k @ self.PHI_k.T + self.UPSILON_k @ self.Q_k @ self.UPSILON_k.T

        self.x_prime_k = x_k_prime
        self.P_prime_k = P_k_prime

    def recurring(self, y_k: list or np.ndarray, *u_k) -> None:
        self.P_prime.append(self.P_prime_k)
        self.x_prime.append(self.x_prime_k)

        self.gain()
        self.K.append(self.K_k)

        self.update_estimation(y_k)
        self.x_hat.append(self.x_hat_k)

        self.update_covariance()
        self.P.append(self.P_k)

        self.propagation(u_k)


class Luenberger:
    # x_hat = x_prime_k + K (y_k - x_prime_k)
    # The α−β−γ filter(sometimes called g - h - k filter) considers constant acceleration, eliminate the lag error.
    # However, the true target dynamic model can also include a jerk (changing acceleration).
    # constant α−β−γ coefficients will produce estimation errors and in some cases lose the target track.
    # This is why we need the detectability.
    # 像是p控制，真值的增长演化和缩小演化会导致存在极大的偏差，可能是稳定的，还可能是危险的。这是模型维度不够导致的。
    # 模型维度不够，即从根本上不满足可检测性的条件。
    # 当维度足够，也即可观测。已知分量可观测，未知分量为0。

    def __init__(self, x0, K, R=1):
        self.K = np.array(K)  # diag(g,h,k) or diag(α−β−γ) but in array
        self.x_hat_k = np.array(x0)
        self.R_k = np.array(R)
        self.x_hat = []
        if np.size(x0) > 1:
            raise TypeError("Do not support currently")

    def recurring(self, y_k: list or np.ndarray, *u_k):
        if not isinstance(y_k, np.ndarray):
            y_k = np.array(y_k)
        self.x_hat.append(self.x_hat_k)
        x_k = self.x_hat_k + self.K * (y_k - self.x_hat_k)
        self.x_hat_k = x_k


# Todo: Actual measurement we do not care about noise, we believe the normal distribution is common.
    # But for simulation, we believe that the real mean is stable, i.e. the estimation is base on mean.
    # This is so strange, if we have two means, the measurement may be one-value or three-value.
    # Because of H, linear Transformation.
    # the (H * True value) and (measurements) are twins, but linear Transformation may be bad.
    # Because we want Bijection, when get X ,then get Y, and if get Y, also get X.
    # Fortunately, the model is still in evolution.
    # 在每一单独时刻不可观测，模型的可观测性能保证演化时间足够长时整个过程的宏观把握。

    # 测量先验： 可能的bug：H坍缩维度， 无发逆推真值
    #          可能的bug：H膨胀维度，得使用最小二乘解法
    # 均值先验： 不容易做不到

def mean_prior(kf, mean: list or np.ndarray, times: int = 1000, *u_k):

    if isinstance(mean, list):
        kf.mean_k = np.array(mean)

    for i in range(times):
        if getattr(kf, 'n', 404) == 1:
            kf.mean_k = kf.PHI_k.dot(kf.mean_k) + rng.normal(0, np.sqrt(kf.Q_k))  # Todo: + kf.GAMMA_k
            if not isinstance(kf.mean_k, np.ndarray):
                kf.mean_k = np.array(kf.mean_k)
            kf.mean.append(kf.mean_k)
        elif getattr(kf, 'n', 404) < 404:
            kf.mean_k = kf.PHI_k.dot(kf.mean_k) + rng.multivariate_normal(np.zeros(kf.n), cov=kf.Q_k)  # Todo: + kf.GAMMA_k
            kf.mean.append(kf.mean_k)
        else:
            pass

        if getattr(kf, 'm', 404) == 1:
            kf.y_k = rng.normal(kf.H_k.dot(kf.mean_k), np.sqrt(kf.R_k), )
        elif getattr(kf, 'm', 404) < 404:
            kf.y_k = rng.multivariate_normal(kf.H_k.dot(kf.mean_k), cov=kf.R_k, )
        elif np.size(kf.x_prime_k, 0) == 1:
            kf.y_k = rng.normal(kf.mean_k, np.sqrt(kf.R_k), )
        else:
            kf.y_k = rng.multivariate_normal(kf.mean_k, cov=kf.R_k, )
        kf.y.append(kf.y_k)
        kf.recurring(kf.y_k, *u_k)
