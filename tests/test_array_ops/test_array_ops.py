import numpy as np
import numpy.testing as npt


def test_get_one_hot():
    from osl_dynamics.array_ops import get_one_hot

    # Case 1: Categorical input
    input_1 = np.array([0, 2, 0, 1])
    output_1 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    npt.assert_equal(get_one_hot(input_1), output_1)

    # Case 2: Categorical input, but input n_states
    input_2 = np.array([0, 2, 0, 1])
    output_2 = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    npt.assert_equal(get_one_hot(input_2, n_states=4), output_2)

    # Case 3: (n_samples, n_states) to be binarized
    input_3 = np.array([
        [0.99, 0.03, 0.03],
        [0.02, 0.02, 0.9],
        [0.80, 0.4, 0.5],
        [-1., -0.5, -1.]
    ])
    output_3 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    npt.assert_equal(get_one_hot(input_3), output_3)

    # Case 4: (n_samples, n_states) to be binarized, input n_states
    input_4 = np.array([
        [0.99, 0.03, 0.03],
        [0.02, 0.02, 0.9],
        [0.80, 0.4, 0.5],
        [-1., -0.5, -1.]
    ])
    output_4 = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    npt.assert_equal(get_one_hot(input_4, n_states=4), output_4)

    # Case 5: list of Categorical input, and input n_states
    input_5_1 = np.array([0, 2, 0, 1])
    output_5_1 = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    input_5_2 = np.array([0, 0, 1])
    output_5_2 = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    input_5 = [input_5_1,input_5_2]
    output_5 = get_one_hot(input_5,n_states=4)
    npt.assert_equal(output_5[0], output_5_1)
    npt.assert_equal(output_5[1],output_5_2)

def test_cov2std():
    from osl_dynamics.array_ops import cov2std

    # Case 1: One covariance matrix
    cov = np.array([[16.0, 4.0], [4.0, 4.0]])
    std = cov2std(cov)
    npt.assert_equal(std, np.array([4.0, 2.0]))

    # Case 2: Two covariance matrices
    covs = np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]])
    stds = cov2std(covs)
    npt.assert_equal(stds, np.array([[4.0, 2.0], [10.0, 20.0]]))


def test_cov2corr():
    from osl_dynamics.array_ops import cov2corr

    # Case 1: One covariance matrix
    cov = np.array([[100.0, -40.0], [-40.0, 400.0]])
    corr = cov2corr(cov)
    npt.assert_equal(corr, np.array([[1.0, -0.2], [-0.2, 1.0]]))

    # Case 2: Two covariance matrices
    covs = np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]])
    corrs = cov2corr(covs)
    npt.assert_equal(corrs, np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, -0.2], [-0.2, 1.0]]]))


def test_stdcorr2cov():
    from osl_dynamics.array_ops import stdcorr2cov
    # Case 1: One covariance matrix, std is a vector
    std = np.array([4.0, 2.0])
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    cov = stdcorr2cov(std, corr)
    npt.assert_equal(cov, np.array([[16.0, 4.0], [4.0, 4.0]]))

    # Case 2: Two covariance matrices, std is two vectors
    stds = np.array([[4.0, 2.0], [10.0, 20.0]])
    corrs = np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, -0.2], [-0.2, 1.0]]])
    covs = stdcorr2cov(stds, corrs)
    npt.assert_equal(covs, np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]]))

    # Case 3: One covariance matrix, std is a diagonal matrix
    std = np.array([[4.0, 0.0], [0.0, 2.0]])
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    cov = stdcorr2cov(std, corr, std_diagonal=True)
    npt.assert_equal(cov, np.array([[16.0, 4.0], [4.0, 4.0]]))

    # Case 4: Two covariance matrices, std is two diagonal matrices
    stds = np.array([[[4.0, 0.0], [0.0, 2.0]], [[10.0, 0.0], [0.0, 20.0]]])
    corrs = np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, -0.2], [-0.2, 1.0]]])
    covs = stdcorr2cov(stds, corrs, std_diagonal=True)
    npt.assert_equal(covs, np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]]))


def test_cov2stdcorr():
    from osl_dynamics.array_ops import cov2stdcorr
    # Case 1: One covariance matrix
    cov = np.array([[16.0, 4.0], [4.0, 4.0]])
    std, corr = cov2stdcorr(cov)
    npt.assert_equal(std, np.array([4.0, 2.0]))
    npt.assert_equal(corr, np.array([[1.0, 0.5], [0.5, 1.0]]))

    # Case 2: Two covariance matrices
    covs = np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]])
    stds, corrs = cov2stdcorr(covs)
    npt.assert_equal(stds, np.array([[4.0, 2.0], [10.0, 20.0]]))
    npt.assert_equal(corrs, np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, -0.2], [-0.2, 1.0]]]))


def test_check_symmetry():
    from osl_dynamics.array_ops import check_symmetry
    # Case 1: One matrix, symmetric
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    npt.assert_equal(check_symmetry(matrix), np.array([True]))

    # Case 2: One matrix, non-symmetric
    matrix = np.array([[1.0, 0.1], [0.0, 1.0]])
    npt.assert_equal(check_symmetry(matrix), np.array([False]))

    # Case 3: Two matrices
    matrix = np.array([[[1.0, 0.1], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    npt.assert_equal(check_symmetry(matrix), np.array([False, True]))

    # Case 4: One matrix, symmetric wrt eps=1e-6
    matrix = np.array([[1.0, 0.99e-6], [0.0, 1.0]])
    npt.assert_equal(check_symmetry(matrix, precision=1e-6), np.array([True]))
    npt.assert_equal(check_symmetry(matrix, precision=1e-7), np.array([False]))


def test_ezclump():
    from osl_dynamics.array_ops import ezclump
    # Test case 1: No clumps (all False)
    arr1 = np.array([False, False, False, False, False])
    expected_result1 = []
    npt.assert_equal(ezclump(arr1), expected_result1)

    # Test case 2: Single clump of True values
    arr2 = np.array([False, True, True, True, False])
    expected_result2 = [slice(1, 4)]
    npt.assert_equal(ezclump(arr2), expected_result2)

    # Test case 3: Multiple clumps of True values
    arr3 = np.array([False, True, False, True, True, False, True, False])
    expected_result3 = [slice(1, 2), slice(3, 5), slice(6, 7)]
    npt.assert_equal(ezclump(arr3), expected_result3)

    # Test case 4: Clumps at both ends
    arr4 = np.array([True, False, False, True, True, False, False, True])
    expected_result4 = [slice(0, 1), slice(3, 5), slice(7, 8)]
    npt.assert_equal(ezclump(arr4), expected_result4)

    # Test case 5: All True values
    arr5 = np.array([True, True, True, True, True])
    expected_result5 = [slice(0, 5)]
    npt.assert_equal(ezclump(arr5), expected_result5)


def test_slice_length():
    from osl_dynamics.array_ops import slice_length

    # Test case 1: Slice with positive values
    s1 = slice(3, 7)
    expected_length1 = 4
    npt.assert_equal(slice_length(s1), expected_length1)

    # Test case 2: Slice with negative values
    s2 = slice(-5, -2)
    expected_length2 = 3
    npt.assert_equal(slice_length(s2), expected_length2)

    # Test case 3: Slice with zero-based start
    s3 = slice(0, 5)
    expected_length3 = 5
    npt.assert_equal(slice_length(s3), expected_length3)

    # Test case 4: Empty slice
    s4 = slice(10, 10)
    expected_length4 = 0
    npt.assert_equal(slice_length(s4), expected_length4)


def test_npz2list():
    from osl_dynamics.array_ops import npz2list
    # Create a list of numpy arrays
    test_data = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

    # Save the npz file
    np.savez('test_data.npz', *test_data)
    # Load the test npz file
    loaded_npz = np.load('test_data.npz')

    # Test npz2list function with the list of numpy arrays
    list_of_arrays = npz2list(loaded_npz)

    # Assert that the returned list contains all arrays from the npz file
    assert len(list_of_arrays) == len(test_data)
    for arr1, arr2 in zip(list_of_arrays, test_data):
        npt.assert_array_equal(arr1, arr2)

    # Clean up the temporary npz file
    import os
    os.remove('test_data.npz')

    x = np.array([1, 2, 3])
    x_list = npz2list(x)
    npt.assert_array_equal(x, x_list[0])


def test_demean_list():
    from osl_dynamics.array_ops import demean_list
    data_1 = [
        [1., 5., 3.],
        [2., 10., 6.],
        [3., 15., 9.]
    ]
    answer_1 = [
        [-1., -5., -3.],
        [0., 0., 0.],
        [1.,5.,3.]
    ]
    npt.assert_allclose(demean_list(data_1), answer_1, atol=1e-8)

    data_2 = [
        [1., 5., np.nan],
        [np.nan, 10., 6.],
        [3., np.nan, 9.]
    ]
    answer_2 = [
        [-1., -2.5],
        [2.5, -1.5],
        [1., 1.5]
    ]
    npt.assert_allclose(demean_list(data_2), answer_2, atol=1e-8)

def test_filter_nan_values():
    from osl_dynamics.array_ops import filter_nan_values
    input = [[np.nan,2,3,np.nan]]
    output = filter_nan_values(input)
    npt.assert_equal(np.array([2,3]),np.array(output[0]))


def test_convert_arrays_to_dtype():
    from osl_dynamics.array_ops import convert_arrays_to_dtype
    # Create input arrays
    arrays_float32 = np.array([1.0, 2.0, 3.0],dtype=np.float32)
    arrays_float64 = [np.array([1.0, 2.0, 3.0],dtype=np.float64), np.array([4.0, 5.0, 6.0],dtype=np.float64)]

    # Expected outputs
    expected_float16 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    expected_float32 = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]

    converted_array = convert_arrays_to_dtype(arrays_float32, np.float16)
    npt.assert_allclose(converted_array, expected_float16)
    assert converted_array.dtype == expected_float16.dtype

    converted_arrays = convert_arrays_to_dtype(arrays_float64, np.float32)
    for i in range(len(converted_arrays)):
        npt.assert_allclose(converted_arrays[i], expected_float32[i])
        assert converted_arrays[i].dtype == expected_float32[i].dtype

def test_estimate_gaussian_distribution():
    from osl_dynamics.array_ops import estimate_gaussian_distribution

    # Generate synthetic data
    np.random.seed(0)
    data = np.array([[1.,-1.],[3.,5.],[5.,8.]])

    # Test case 1: nonzero_means=True, keepdims=True
    mean, cov = estimate_gaussian_distribution(data, nonzero_means=True, keepdims=True)
    npt.assert_almost_equal(mean, np.array([[3.,4.]]))
    npt.assert_almost_equal(cov, np.array([[[8/3,6.],[6.,14.]]]))

    # Test case 2: nonzero_means=False, keepdims=True
    mean, cov = estimate_gaussian_distribution(data, nonzero_means=False, keepdims=True)
    npt.assert_almost_equal(mean, np.array([[0.,0.]]))
    npt.assert_almost_equal(cov,np.array([[[8/3,6.],[6.,14.]]]))

    # Test case 3: nonzero_means=True, keepdims=False
    mean, cov = estimate_gaussian_distribution(data, nonzero_means=True, keepdims=False)
    npt.assert_almost_equal(mean,np.array([3.,4.]))
    npt.assert_almost_equal(cov, np.array([[8/3,6.],[6.,14.]]))

    # Test case 4: nonzero_means=False, keepdims=False
    mean, cov = estimate_gaussian_distribution(data, nonzero_means=False, keepdims=False)
    npt.assert_almost_equal(mean, np.array([0.,0.]))
    npt.assert_almost_equal(cov, np.array([[8/3,6.],[6.,14.]]))

