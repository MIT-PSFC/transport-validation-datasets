"""The zk kernel's analytic derivatives against central differences.

mkgp's optimizer climbs the LML along the hyperparameter derivatives,
and the gradient anchors and the fitted gradients go through the input derivatives.
"""

import numpy as np
import pytest

from transport_validation_datasets.gp_fitting.zk.kernel import build_kernel

PEDESTAL_RHO = 0.95
HYPS = np.array([1.3, 0.6, 0.2, 0.08])  # var, l1, l2, lw
# Both grids cross the transition, where every derivative is far from zero
X1 = np.linspace(0.5, 1.2, 9)
X2 = np.linspace(0.55, 1.25, 7)
STEP = 1e-6


# der 0 is the covariance, -1 and +1 its derivative in x1 and in x2, and 2 in both
@pytest.mark.parametrize("der", [0, -1, 1, 2])
@pytest.mark.parametrize("hder", [0, 1, 2, 3])
def test_hyperparameter_derivatives(der, hder):
    hyps_up = HYPS.copy()
    hyps_up[hder] += STEP
    hyps_down = HYPS.copy()
    hyps_down[hder] -= STEP
    cov_up = build_kernel(PEDESTAL_RHO, hyps_up)(X1, X2, der=der)
    cov_down = build_kernel(PEDESTAL_RHO, hyps_down)(X1, X2, der=der)
    central_difference = (cov_up - cov_down) / (2 * STEP)

    analytic = build_kernel(PEDESTAL_RHO, HYPS)(X1, X2, der=der, hder=hder)

    np.testing.assert_allclose(analytic, central_difference, rtol=1e-6, atol=1e-6)


def test_input_derivatives():
    kernel = build_kernel(PEDESTAL_RHO, HYPS)
    d_dx1 = (kernel(X1 + STEP, X2) - kernel(X1 - STEP, X2)) / (2 * STEP)
    d_dx2 = (kernel(X1, X2 + STEP) - kernel(X1, X2 - STEP)) / (2 * STEP)
    d_dx1_up = kernel(X1, X2 + STEP, der=-1)
    d_dx1_down = kernel(X1, X2 - STEP, der=-1)
    d2_dx1dx2 = (d_dx1_up - d_dx1_down) / (2 * STEP)

    np.testing.assert_allclose(kernel(X1, X2, der=-1), d_dx1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(kernel(X1, X2, der=1), d_dx2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(kernel(X1, X2, der=2), d2_dx1dx2, rtol=1e-6, atol=1e-6)
