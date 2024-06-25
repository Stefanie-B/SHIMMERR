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


@pytest.mark.parametrize(
    "args, expected, expected_raises",
    [
        (
            [[[0, 0, 0], [1, 1, 1]], [0, 0, 1], 2],
            [[0.5, 0.5, 0.5], [0, 0, 1], 2, [0, 0, 0], 1],
            None,
        ),
        (
            [[[0, 4j, 0], [1, 1, 1]], [0, 0, 1], 2],
            [[0.5, 0.5, 0.5], [0, 0, 1], 2, [0, 0, 0], 1],
            TypeError,
        ),
        (
            [[[0, 0, 0], [1, 1, 1], [-7, -10, -1]], [0, 0, 1], 2],
            [[-2, -3, 0], [0, 0, 1], 2, [0, 0, 0], 1],
            None,
        ),
        (
            [[[0, 0, 0], [1, 1, 1], [-7, -10, -1]], [0, 0, 1]],
            [[-2, -3, 0], [0, 0, 1], 1, [0, 0, 0], 1 + 0j],
            None,
        ),
        (
            [[[0, 0, 0, 0], [1, 1, 1], [-7, -10, -1]], [0, 0, 1]],
            [[-2, -3, 0], [0, 0, 1], 1, [0, 0, 0], 1],
            ValueError,
        ),
    ],
)
def test_Tile_init(args, expected, expected_raises):
    from beam_errors.array import Tile

    if expected_raises is not None:
        with pytest.raises(expected_raises):
            test_tile = Tile(*args)
    else:
        test_tile = Tile(*args)
        npt.assert_almost_equal(test_tile.p, expected[0], 5)
        npt.assert_equal(test_tile.d, expected[1])
        npt.assert_equal(test_tile.g, expected[2])
        npt.assert_equal(test_tile.elements[0].p, expected[3])
        npt.assert_array_almost_equal(test_tile.elements[0].g, expected[4], 5)


@pytest.mark.parametrize(
    "args_init, args_response, expected",
    [
        (
            [[[-1, -1, 0], [1, 1, 0]], [0, 0, 1], 2],
            [[0, 0, 1], 150e6, "omnidirectional", 2],
            2,
        ),
    ],
)
def test_Tile_response(args_init, args_response, expected):
    from beam_errors.array import Tile

    test_tile = Tile(*args_init)
    npt.assert_almost_equal(test_tile.calculate_response(*args_response), expected, 5)


# station init --> add pointing check (unit vector)
# calculate response
