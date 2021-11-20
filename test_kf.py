from kf import KF, Luenberger, mean_prior
import numpy as np
from unittest2 import TestCase
rng = np.random.default_rng()


class TestKF(TestCase):
    def test_singe_scalar(self):
        kf = KF(x0=[1], P0=[0.3], H=[1], Q=[0], R=[1], PHI=[1], GAMMA=[1], UPSILON=[1])
        kf.recurring([0.1])
        for i in range(1000):
            kf.recurring(0.1 + 0.01*rng.standard_normal(1))
        x_hat = kf.x_hat_k[-1]
        self.assertAlmostEqual(0.1001, 0.1, places=2)
        self.assertAlmostEqual(x_hat, 0.1, places=2)

    def test_luenberger(self):
        kk = 0.3
        k = Luenberger(x0=1, K=kk, R=1)
        for i in range(100):
            k.recurring(0.1 + 0.001*rng.standard_normal(1))
            # k.recurring(0.1)
        x_hat = k.x_hat_k[0]
        self.assertAlmostEqual(x_hat, 0.1, places=2)

    def test_singe_scalar_two_measurement(self):
        Hs = np.array([[1, 1]]).T
        Rs = np.eye(2)
        kf = KF(x0=[1], P0=[0.3], H=Hs, Q=[0], R=Rs, PHI=[1], GAMMA=[1], UPSILON=[1])
        ys = np.array([0.1, 0.1])
        kf.recurring(ys)
        for i in range(10000):
            kf.recurring(ys, 0)
        x_hat = kf.x_hat_k[-1]
        self.assertAlmostEqual(x_hat, 0.1, places=3)

    def test_two_scalar_one_measurement(self):
        x0 = np.array([1, 1])
        Ps = np.array([[2, 0], [0, 0]])
        Qs = np.eye(2)
        PHIs = np.eye(2)
        GAMMAs = np.eye(2)
        UPSILONs = np.eye(2)
        Hs = np.array([[1, 0]])
        kf = KF(x0=x0, P0=Ps, H=Hs, Q=Qs, R=[1], PHI=PHIs, GAMMA=GAMMAs, UPSILON=UPSILONs)
        # kf.recurring([0.1])
        for i in range(10):
            kf.recurring([0.1])
        x_hat = kf.x_hat_k[0]
        self.assertAlmostEqual(x_hat, 0.1, places=3)

    def test_2x_2y(self):
        x0 = np.array([1, 0])
        Ps = np.array([[2, 0], [0, 2]])
        Qs = np.zeros((2, 2))
        PHIs = np.eye(2)
        GAMMAs = np.eye(2)
        UPSILONs = np.eye(2)
        Hs = np.array([[1, -0.5], [1, 1]])
        Rs = np.eye(2)
        kf = KF(x0=x0, P0=Ps, H=Hs, Q=Qs, R=Rs, PHI=PHIs, GAMMA=GAMMAs, UPSILON=UPSILONs)
        ys = np.array([0.1, 0.1])
        # kf.recurring(ys)
        for i in range(10000):
            kf.recurring(ys)
        x_hat = kf.x_hat_k[0]
        self.assertAlmostEqual(x_hat, 0.1, places=3)

    def test_my_homework2(self):
        x0 = np.array([1, 1])
        Ps = np.array([[2, 0], [0, 1]])
        Qs = np.array([[4.9502e-5, 0], [0, 9.8510e-3]])
        Qs = Qs * Qs
        PHIs = np.array([[9.9985e-1, 9.8510e-3], [-2.9553e-2, 9.7030e-1]])
        GAMMAs = np.eye(2)
        UPSILONs = np.eye(2)
        Hs = np.array([[1, 0]])
        kf = KF(x0=x0, P0=Ps, H=Hs, Q=Qs, R=[0.01], PHI=PHIs, GAMMA=GAMMAs, UPSILON=UPSILONs)
        # kf.recurring([0.1])
        for i in range(100):
            kf.recurring([0.1])
        x_hat = kf.x_hat_k[0]
        self.assertAlmostEqual(x_hat, 0.1, places=2)

    def test_my_homework3(self):
        x0 = np.array([1, 1])
        Ps = np.array([[2, 0], [0, 1]])
        Qs = np.array([[4.9502e-5, 0], [0, 9.8510e-3]])
        Qs = Qs * Qs
        PHIs = np.array([[9.9985e-1, 9.8510e-3], [-2.9553e-2, 9.7030e-1]])
        GAMMAs = np.eye(2)
        UPSILONs = np.eye(2)
        Hs = np.array([[1, 0], [0, 1], [1, 1]])
        Rs = 0.01 * np.eye(3)
        kf = KF(x0=x0, P0=Ps, H=Hs, Q=Qs, R=Rs, PHI=PHIs, GAMMA=GAMMAs, UPSILON=UPSILONs)
        ys = np.array([0.1, 0.1, 0.18])
        # kf.recurring(ys)
        for i in range(1000):
            kf.recurring(ys)
        x_hat = kf.x_prime_k[0]
        self.assertAlmostEqual(x_hat, 0.1, places=2)
        self.assertAlmostEqual(kf.x_prime_k[1], 0.08, places=2)

    def test_duck(self):
        x0 = np.array([1, 1])
        Ps = np.array([[2, 0], [0, 1]])
        Qs = np.array([[4.9502e-5, 0], [0, 9.8510e-3]])
        Qs = Qs * Qs
        # PHIs = np.array([[9.9985e-1, 9.8510e-3], [-2.9553e-2, 9.7030e-1]])
        PHIs = np.array([[1, 0], [0, 1]])
        GAMMAs = np.eye(2)
        UPSILONs = np.eye(2)
        Hs = np.array([[1, 0]])
        kf = KF(x0=x0, P0=Ps, H=Hs, Q=Qs, R=[0.01], PHI=PHIs, GAMMA=GAMMAs, UPSILON=UPSILONs)
        mean_prior(kf, [2, 2])
        x_hat = kf.x_hat_k[0]
        self.assertAlmostEqual(x_hat, kf.mean_k[0], places=2)
        # self.assertAlmostEqual(kf.x_hat_k[1], kf.mean_k[1], places=2)

    def test_bird(self):
        kf = KF(x0=[1], P0=[0.3], H=[1], Q=[0], R=[1], PHI=[1], GAMMA=[1], UPSILON=[1])
        mean_prior(kf, [0], times=100)
        x_hat = kf.x_hat_k[-1]
        self.assertAlmostEqual(x_hat, .0, places=4)
