import numpy as np
import numpy.testing as npt


def test_stdcor2cov():
    from rotation.utils import stdcor2cov
    stds = np.array([[4.0,2.0],[10.0,20.0]])
    corrs = np.array([[[1.0,0.5],[0.5,1.0]],[[1.0,-0.2],[-0.2,1.0]]])
    covs = stdcor2cov(stds, corrs)
    npt.assert_equal(covs,np.array([[[16.0,4.0],[4.0,4.0]],[[100.0,-40.0],[-40.0,400.0]]]))

    stds = np.array([[[4.0,0.0],[0.0,2.0]],[[10.0,0.0],[0.0,20.0]]])
    covs = stdcor2cov(stds,corrs)
    npt.assert_equal(covs, np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]]))

def test_first_eigenvector():
    from rotation.utils import first_eigenvector
    matrix = np.array([[2.0,0.0],[0.0,1.0]])
    first_eigen = first_eigenvector(matrix)
    npt.assert_equal(first_eigen,np.array([1.0,0.0]))