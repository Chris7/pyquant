__author__ = 'chris'
import unittest
from random import randint, random

import numpy as np
from six.moves import xrange
from sympy import symbols, diff, exp, Piecewise

from pyquant.tests.mixins import GaussianMixin
from pyquant import peaks
from pyquant import cpeaks

def get_gauss_value(x, amp, mu, std):
    return amp*np.exp(-(x - mu)**2/(2*std**2))

class MathTests(GaussianMixin, unittest.TestCase):
    def setUp(self):
        super(MathTests, self).setUp()
        self.std_2, self.std2_2 = 0.5, 0.75
        self.one_bigauss_params = np.array([self.amp, self.mu, self.std, self.std_2], dtype=np.float)
        self.two_bigauss_params = np.array([self.amp, self.mu, self.std, self.std_2, self.amp, self.mu2, self.std, self.std2_2], dtype=np.float)
        self.one_bigauss = peaks.bigauss_ndim(self.x, self.one_bigauss_params)
        self.two_bigauss = peaks.bigauss_ndim(self.x, self.two_bigauss_params)

    def test_jacobians(self):
        one_gauss_jac = peaks.gauss_jac(self.one_gauss_params, self.x, self.one_gauss, False)
        self.assertEqual(one_gauss_jac.tolist(), np.zeros_like(self.one_gauss_params).tolist())

        two_gauss_jac = peaks.gauss_jac(self.two_gauss_params, self.x, self.two_gauss, False)
        self.assertEqual(two_gauss_jac.tolist(), np.zeros_like(self.two_gauss_params).tolist())
        one_bigauss_jac = peaks.bigauss_jac(self.one_bigauss_params, self.x, self.one_bigauss, False)
        self.assertEqual(one_bigauss_jac.tolist(), np.zeros_like(self.one_bigauss_params).tolist())

        two_bigauss_jac = peaks.bigauss_jac(self.two_bigauss_params, self.x, self.two_bigauss, False)
        self.assertEqual(two_bigauss_jac.tolist(), np.zeros_like(self.two_bigauss_params).tolist())
        y, x, a, u, s1, s1_2, a2, u2, s2, s2_2, a3, u3, s3, s3_2 = symbols('y x a u s1 s1_2 a2 u2 s2 s2_2 a3 u3 s3 s3_2')
        three_gauss = (y - (a * exp(-(u - x) ** 2 / (2 * s1 ** 2)) + a2 * exp(-(u2 - x) ** 2 / (2 * s2 ** 2)) + a3 * exp(-(u3 - x) ** 2 / (2 * s3 ** 2)))) ** 2
        three_gauss2 = (y - (a * exp(-(u - x) ** 2 / (2 * s1_2 ** 2)) + a2 * exp(-(u2 - x) ** 2 / (2 * s2_2 ** 2)) + a3 * exp(-(u3 - x) ** 2 / (2 * s3_2 ** 2)))) ** 2
        bigauss = Piecewise((three_gauss, x<u))
        deriv_store = {}
        for i in xrange(2):
            subs = [
                ('a', random()),
                ('u', randint(1,10)),
                ('s1', random()),
                ('a2', random()),
                ('u2', randint(12, 20)),
                ('s2', random()),
                ('a3', random()),
                ('u3', randint(22, 30)),
                ('s3', random()),
            ]
            params = np.array([i[1] for i in subs], dtype=float)
            noisy_params = params + 2*np.random.rand(params.shape[0])
            gauss_x = np.linspace(-10, 40, 100)
            gauss_y = peaks.gauss_ndim(gauss_x, noisy_params)
            jacobian = peaks.gauss_jac(params, gauss_x, gauss_y, False)
            for var_index, var in enumerate([a,u,s1,a2,u2,s2,a3,u3,s3]):
                deriv = deriv_store.setdefault(var, diff(three_gauss, var))
                pq_jac = jacobian[var_index]
                sympy_jacobian = sum([deriv.subs(dict(subs, **{'x': xi, 'y': yi})) for xi, yi in zip(gauss_x, gauss_y)])
                np.testing.assert_allclose(pq_jac, np.array(sympy_jacobian, dtype=float),
                                           err_msg='d{} - pq: {}, sympy: {}'.format(var, pq_jac,
                                                                                       sympy_jacobian), atol=1e-4)

    def test_hessians(self):
        y, x, a, u, s1, a2, u2, s2, a3, u3, s3 = symbols('y x a u s1 a2 u2 s2 a3 u3 s3')
        three_gauss = (y - (
        a * exp(-(u - x) ** 2 / (2 * s1 ** 2)) + a2 * exp(-(u2 - x) ** 2 / (2 * s2 ** 2)) + a3 * exp(
            -(u3 - x) ** 2 / (2 * s3 ** 2)))) ** 2
        hess_store = {}
        for _ in xrange(2):
            subs = [
                ('a', random()+0.5),
                ('u', random()*10),
                ('s1', random()*3+2),
                ('a2', random()+0.5),
                ('u2', random()*10+10),
                ('s2', random()*3+2),
                ('a3', random()+0.5),
                ('u3', random()*10+20),
                ('s3', random()*3+2),
            ]
            params = np.array([i[1] for i in subs], dtype=float)
            noisy_params = params + 2 * np.random.rand(params.shape[0])
            gauss_x = np.linspace(-10, 40, 100)
            gauss_y = cpeaks.gauss_ndim(gauss_x, noisy_params)
            hessian = cpeaks.gauss_hess(params, gauss_x, gauss_y)
            for var_index, var in enumerate([a, u, s1, a2, u2, s2, a3, u3, s3]):
                for var_index2, var2 in enumerate([a, u, s1, a2, u2, s2, a3, u3, s3]):
                    deriv = hess_store.setdefault((var, var2), diff(three_gauss, var, var2))
                    sympy_hessian = sum([deriv.subs(dict(subs, **{'x': xi, 'y': yi})) for xi, yi in zip(gauss_x, gauss_y)])
                    pq_hess = hessian[var_index, var_index2]
                    np.testing.assert_allclose(pq_hess, np.array(sympy_hessian, dtype=float), err_msg='d{}d{} - pq: {}, sympy: {}'.format(var, var2, pq_hess, sympy_hessian), atol=1e-4)


if __name__ == '__main__':
    unittest.main()
