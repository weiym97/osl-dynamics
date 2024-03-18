import numpy as np
import numpy.testing as npt

def test_BICVkmeans():
    from osl_dynamics.evaluate.cross_validation import BICVkmeans

    cv = BICVkmeans(n_clusters=2,n_samples=8,n_channels=2,partition_rows=4,partition_columns=2)

    # Step 1: Check partition_indices
    cv.partition_indices()

    # Step 2: Check the fold specification
    cv.row_indices = [np.array([7,4]),
                      np.array([5,0]),
                      np.array([6,3]),
                      np.array([2,1])]
    cv.column_indices = [np.array([1]),
                         np.array([0])]
    npt.assert_equal(cv.fold_indices(0,0),
                    (
                        np.array([5,0,6,3,2,1]),np.array([7,4]),
                        np.array([0]),np.array([1])
                    )
                    )
    npt.assert_equal(cv.fold_indices(2, 1),
                     (
                         np.array([7, 4, 5, 0, 2, 1]), np.array([6, 3]),
                         np.array([1]), np.array([0])
                     )
                     )

    # Fix the train/test rows and X/Y columns now.
    row_train = np.array([3,2,1,0])
    row_test = np.array([7,6,5,4])
    column_X = np.array([0])
    column_Y = np.array([1])

    # Step 3: Check the Y_train step
    data = np.array([[0.1,1.0],
                       [100.,1.0],
                       [-1000.,-1.0],
                       [0.0,-1.0],
                       [0.1, 0.5],
                       [100., 0.75],
                       [-1000., -0.25],
                       [0.0, -1.0]]
    )
    spatial_Y_train, temporal_Y_train = cv.Y_train(data,row_train,column_Y)
    npt.assert_equal(spatial_Y_train**2,np.array([[1.],[1.]]))
    npt.assert_equal(spatial_Y_train[temporal_Y_train], data[row_train][:,column_Y])

    # Step 4: Check the X_train step
    temporal_Y_train = np.array([1,1,0,0])
    spatial_X_train = cv.X_train(data, row_train, column_X, temporal_Y_train)
    npt.assert_equal(spatial_X_train,np.array([[50.05],[-500.]]))

    # Step 5: Check the X_test step
    spatial_X_train = np.array([[51.],[-49.]])
    temporal_X_test = cv.X_test(data,row_test,column_X,spatial_X_train)
    npt.assert_equal(temporal_X_test,np.array([1,1,0,1]))

    # Step 6: Cehck the Y_test step
    temporal_X_test = np.array([1,0,1,0])
    spatial_Y_train = np.array([[0.5],[-0.5]])
    metric = cv.Y_test(data,row_test,column_Y,temporal_X_test,spatial_Y_train)
    npt.assert_equal(metric,np.array([0.5**2+1.25**2+0.75**2])/4)