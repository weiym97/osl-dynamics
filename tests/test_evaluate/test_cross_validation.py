import os
import shutil

import numpy as np
import numpy.testing as npt


def test_CVSplit():
    from osl_dynamics.evaluate.cross_validation import CVSplit

    save_dir = "./test_cvsplit/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    try:
        # Case 1: row ShuffleSplit, column KFold, combination
        config_1 = {
            "split_row": {
                "n_samples": 10,
                "method": "ShuffleSplit",
                "method_kwargs": {
                    "n_splits": 5,
                    "train_size": 0.8,
                },
            },
            "split_column": {
                "n_samples": 4,
                "method": "KFold",
                "method_kwargs": {
                    "n_splits": 2,
                },
            },
            "strategy": "combination",
        }
        cv_splitter_1 = CVSplit(**config_1)
        assert cv_splitter_1.get_n_splits() == 10  # 5 * 2

        for row_train, row_test, col_train, col_test in cv_splitter_1.split():
            assert len(row_train) == 8
            assert len(row_test) == 2
            assert len(col_train) + len(col_test) == 4

        # Case 2: row ShuffleSplit, column KFold, pairing
        config_2 = {
            "split_row": {
                "n_samples": 5,
                "method": "ShuffleSplit",
                "method_kwargs": {
                    "n_splits": 3,
                    "train_size": 0.8,
                },
            },
            "split_column": {
                "n_samples": 4,
                "method": "KFold",
                "method_kwargs": {
                    "n_splits": 2,
                },
            },
            "strategy": "pairing",
        }
        cv_splitter_2 = CVSplit(**config_2)
        assert cv_splitter_2.get_n_splits() == 2  # min(3, 2)

        # Case 3: row KFold, column ShuffleSplit, combination
        config_3 = {
            "split_row": {
                "n_samples": 5,
                "method": "KFold",
                "method_kwargs": {
                    "n_splits": 5,
                },
            },
            "split_column": {
                "n_samples": 4,
                "method": "ShuffleSplit",
                "method_kwargs": {
                    "n_splits": 2,
                    "train_size": 0.5,
                },
            },
            "strategy": "combination",
        }
        cv_splitter_3 = CVSplit(**config_3)
        assert cv_splitter_3.get_n_splits() == 10  # 5 * 2

        # Case 4: row KFold, column ShuffleSplit, pairing
        config_4 = {
            "split_row": {
                "n_samples": 5,
                "method": "KFold",
                "method_kwargs": {
                    "n_splits": 5,
                },
            },
            "split_column": {
                "n_samples": 4,
                "method": "ShuffleSplit",
                "method_kwargs": {
                    "n_splits": 2,
                    "train_size": 0.5,
                },
            },
            "strategy": "pairing",
        }
        cv_splitter_4 = CVSplit(**config_4)
        assert cv_splitter_4.get_n_splits() == 2  # min(5, 2)

        # Case 5: row ShuffleSplit only (naive cross-validation)
        config_5 = {
            "split_row": {
                "n_samples": 5,
                "method": "ShuffleSplit",
                "method_kwargs": {
                    "n_splits": 3,
                    "train_size": 0.8,
                },
            },
        }
        cv_splitter_5 = CVSplit(**config_5)
        assert cv_splitter_5.get_n_splits() == 3

        # Case 6: row KFold only
        config_6 = {
            "split_row": {
                "n_samples": 5,
                "method": "KFold",
                "method_kwargs": {
                    "n_splits": 5,
                },
            },
        }
        cv_splitter_6 = CVSplit(**config_6)
        assert cv_splitter_6.get_n_splits() == 5

        # Test saving
        save_dir_1 = f"{save_dir}/case_1/"
        os.makedirs(save_dir_1, exist_ok=True)
        save_dir_5 = f"{save_dir}/case_5/"
        os.makedirs(save_dir_5, exist_ok=True)
        cv_splitter_1.save(save_dir_1)
        cv_splitter_5.save(save_dir_5)

        # Verify saved files exist
        assert os.path.exists(f"{save_dir_1}/fold_indices_1.json")
        assert os.path.exists(f"{save_dir_5}/fold_indices_1.json")

    finally:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)


def test_partition_indices():
    from osl_dynamics.evaluate.cross_validation import CVBase

    save_dir = "./test_tmp_partition_indices/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    try:
        os.makedirs(f"{save_dir}case_1/", exist_ok=True)
        os.makedirs(f"{save_dir}case_2/", exist_ok=True)

        n_samples = 1000
        n_channels = 50

        # Case 1: Default settings
        cv_1 = CVBase(
            n_samples=n_samples,
            n_channels=n_channels,
            save_dir=f"{save_dir}case_1/",
        )
        row_indices = np.load(
            os.path.join(f"{save_dir}case_1/", "row_indices.npz")
        )
        column_indices = np.load(
            os.path.join(f"{save_dir}case_1/", "column_indices.npz")
        )
        row_indices = np.sort(
            np.concatenate([row_indices[key] for key in row_indices.keys()])
        )
        column_indices = np.sort(
            np.concatenate(
                [column_indices[key] for key in column_indices.keys()]
            )
        )
        npt.assert_array_equal(row_indices, np.arange(n_samples))
        npt.assert_array_equal(column_indices, np.arange(n_channels))

        # Case 2: Multi-folds
        cv_2 = CVBase(
            n_samples=n_samples,
            n_channels=n_channels,
            save_dir=f"{save_dir}case_2/",
            partition_rows=7,
            partition_columns=9,
        )
        row_indices = np.load(
            os.path.join(f"{save_dir}case_2/", "row_indices.npz")
        )
        column_indices = np.load(
            os.path.join(f"{save_dir}case_2/", "column_indices.npz")
        )
        row_indices = np.sort(
            np.concatenate([row_indices[key] for key in row_indices.keys()])
        )
        column_indices = np.sort(
            np.concatenate(
                [column_indices[key] for key in column_indices.keys()]
            )
        )
        npt.assert_array_equal(row_indices, np.arange(n_samples))
        npt.assert_array_equal(column_indices, np.arange(n_channels))

        # Case 3: Use the results from case 1
        cv_3 = CVBase(
            n_samples=n_samples,
            n_channels=n_channels,
            row_indices=f"{save_dir}case_1/row_indices.npz",
            column_indices=f"{save_dir}case_1/column_indices.npz",
        )

        npt.assert_array_equal(cv_1.row_indices[0], cv_3.row_indices[0])
        npt.assert_array_equal(cv_1.row_indices[1], cv_3.row_indices[1])
        npt.assert_array_equal(
            cv_1.column_indices[0], cv_3.column_indices[0]
        )
        npt.assert_array_equal(
            cv_1.column_indices[1], cv_3.column_indices[1]
        )

    finally:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)


def test_fold_indices():
    from osl_dynamics.evaluate.cross_validation import CVBase

    n_samples = 7
    n_channels = 5
    cv = CVBase(
        n_samples=n_samples, n_channels=n_channels, partition_rows=3
    )
    cv.row_indices = [
        np.array([6, 2]),
        np.array([4, 0]),
        np.array([5, 3, 1]),
    ]
    cv.column_indices = [np.array([4, 2, 0]), np.array([3, 1])]

    # Fold (1,1)
    row_train, row_test, column_X, column_Y = cv.fold_indices(0, 0)
    npt.assert_array_equal(row_train, np.array([0, 1, 3, 4, 5]))
    npt.assert_array_equal(row_test, np.array([2, 6]))
    npt.assert_array_equal(column_X, np.array([1, 3]))
    npt.assert_array_equal(column_Y, np.array([0, 2, 4]))

    # Fold (1,2)
    row_train, row_test, column_X, column_Y = cv.fold_indices(0, 1)
    npt.assert_array_equal(row_train, np.array([0, 1, 3, 4, 5]))
    npt.assert_array_equal(row_test, np.array([2, 6]))
    npt.assert_array_equal(column_X, np.array([0, 2, 4]))
    npt.assert_array_equal(column_Y, np.array([1, 3]))

    # Fold (2,1)
    row_train, row_test, column_X, column_Y = cv.fold_indices(1, 0)
    npt.assert_array_equal(row_train, np.array([1, 2, 3, 5, 6]))
    npt.assert_array_equal(row_test, np.array([0, 4]))
    npt.assert_array_equal(column_X, np.array([1, 3]))
    npt.assert_array_equal(column_Y, np.array([0, 2, 4]))

    # Fold (2,2)
    row_train, row_test, column_X, column_Y = cv.fold_indices(1, 1)
    npt.assert_array_equal(row_train, np.array([1, 2, 3, 5, 6]))
    npt.assert_array_equal(row_test, np.array([0, 4]))
    npt.assert_array_equal(column_X, np.array([0, 2, 4]))
    npt.assert_array_equal(column_Y, np.array([1, 3]))

    # Fold (3,1)
    row_train, row_test, column_X, column_Y = cv.fold_indices(2, 0)
    npt.assert_array_equal(row_train, np.array([0, 2, 4, 6]))
    npt.assert_array_equal(row_test, np.array([1, 3, 5]))
    npt.assert_array_equal(column_X, np.array([1, 3]))
    npt.assert_array_equal(column_Y, np.array([0, 2, 4]))

    # Fold (3,2)
    row_train, row_test, column_X, column_Y = cv.fold_indices(2, 1)
    npt.assert_array_equal(row_train, np.array([0, 2, 4, 6]))
    npt.assert_array_equal(row_test, np.array([1, 3, 5]))
    npt.assert_array_equal(column_X, np.array([0, 2, 4]))
    npt.assert_array_equal(column_Y, np.array([1, 3]))
