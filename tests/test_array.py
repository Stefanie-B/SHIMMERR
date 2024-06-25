import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "args, expected, expected_raises",
    [
        ([[0, 0, 0], 1], [[0, 0, 0], 1], None),
        ([[0, 0j, 0], 1], [], TypeError),
        ([[0, 0, "foobar"], 1], [], TypeError),
        ([[0, 0, 4, 3], 1], [], ValueError),
        ([np.array([0, 0, 4]), 1], [[0, 0, 4], 1], None),
        ([np.array([0, 8, 4]), 1j], [[0, 8, 4], 1j], None),
        ([np.array([0, 0, 4]), 2.3 + 6.3j], [[0, 0, 4], 2.3 + 6.3j], None),
        ([np.array([0, 0, 4]), "foobar"], [], TypeError),
        ([[34, -6, 4]], [[34, -6, 4], 1 + 0j], None),
    ],
)
def test_Antenna_init(args, expected, expected_raises):
    from beam_errors.array import Antenna

    if expected_raises is not None:
        with pytest.raises(expected_raises):
            test_antenna = Antenna(*args)
    else:
        test_antenna = Antenna(*args)
        npt.assert_equal(test_antenna.p, expected[0])
        npt.assert_equal(test_antenna.g, expected[1])


@pytest.mark.parametrize(
    "args_init, args_response, expected",
    [
        ([[0, 0, 0], 1], [[0, 0, 1], 150e6, "omnidirectional"], 1),
        ([[0, 0, 0], 1], [[0, 0, 1], 150e6], 1),
        ([[0, 0, 0], 5j], [[0, 0, 1], 150e6], 5j),
        ([[0, 0, 0], 5j], [[0, 1, 0], 150e6], 5j),
    ],
)
def test_Antenna_response(args_init, args_response, expected):
    from beam_errors.array import Antenna

    test_antenna = Antenna(*args_init)
    npt.assert_equal(test_antenna.calculate_response(*args_response), expected)
