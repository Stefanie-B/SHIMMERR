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
        (
            [[0, 0, 0], 5j],
            [[0, 1 / np.sqrt(2), 1 / np.sqrt(2)], 150e6, "simplified"],
            5j * np.exp(-2 * (np.pi / 4) ** 3),
        ),
    ],
)
def test_Antenna_response(args_init, args_response, expected):
    from beam_errors.array import Antenna

    test_antenna = Antenna(*args_init)
    npt.assert_almost_equal(test_antenna.calculate_response(*args_response), expected)


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
            [[0, 0, 1], 150e6, None],
            2,
        ),
        (
            [[[-1, 0, 0], [1, 0, 0]], [0, np.sqrt(2) / 2, np.sqrt(2) / 2]],
            [[0, 0, 1], 150e6],
            1,
        ),
        (
            [[[-1, 0, 0], [1, 0, 0]], [0, 0, 1]],
            [[np.sqrt(2) / 2, 0, np.sqrt(2) / 2], 150e6, 1],
            (
                np.exp(1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
                + np.exp(-1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
            )
            / 2,
        ),
        (
            [[[-1, 0, 0], [1, 0, 0]], [np.sqrt(2) / 2, 0, np.sqrt(2) / 2]],
            [[0, 0, 1], 150e6, 2],
            (
                2 * np.exp(1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
                + 2 * np.exp(-1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
            )
            / 2,
        ),
    ],
)
def test_Tile_response(args_init, args_response, expected):
    from beam_errors.array import Tile

    test_tile = Tile(*args_init)
    test_tile.set_ENU_positions(
        np.eye(3), np.array([0, 0, 0])
    )  # Since the tile is not part of a station it doesn't have an ENU position yet
    npt.assert_almost_equal(test_tile.calculate_response(*args_response), expected, 5)


@pytest.mark.parametrize(
    "args, expected, expected_raises",
    [
        (
            [[[[0, 0, 0], [1, 1, 1]]], [0, 0, 1], 2],
            [[0.5, 0.5, 0.5], [0, 0, 1], 2, [0.5, 0.5, 0.5], 1, [0, 0, 0], 1],
            None,
        ),
        (
            [[[[0, 4j, 0], [1, 1, 1]]], [0, 0, 1], 2],
            [],
            TypeError,
        ),
        (
            [[[[0, 0, 0], [1, 1, 1]], [[-6.5, -10.5, -1.5]]], [0, 0, 1], 2],
            [[-3, -5, -0.5], [0, 0, 1], 2, [0.5, 0.5, 0.5], 1, [0, 0, 0], 1],
            None,
        ),
        (
            [[[[0, 0, 0]], [[1, 1, 1], [-7, -10, -1]]], [0, 1, 0]],
            [[-1.5, -2.25, 0], [0, 1, 0], 1, [0, 0, 0], 1 + 0j, [0, 0, 0], 1 + 0j],
            None,
        ),
        (
            [[[[0, 0, 0, 0], [1, 1, 1], [-7, -10, -1]]], [0, 0, 1]],
            [],
            ValueError,
        ),
    ],
)
def test_Station_init(args, expected, expected_raises):
    from beam_errors.array import Station

    if expected_raises is not None:
        with pytest.raises(expected_raises):
            test_station = Station(*args)
    else:
        test_station = Station(*args)
        npt.assert_almost_equal(test_station.p, expected[0], 5)
        npt.assert_equal(test_station.d, expected[1])
        npt.assert_equal(test_station.g, expected[2])
        npt.assert_equal(test_station.elements[0].p, expected[3])
        npt.assert_array_almost_equal(test_station.elements[0].g, expected[4], 5)
        npt.assert_equal(test_station.elements[0].elements[0].p, expected[5])
        npt.assert_array_almost_equal(
            test_station.elements[0].elements[0].g, expected[6], 5
        )


@pytest.mark.parametrize(
    "args_init, args_response, expected",
    [
        (
            [[[[-1.5, -1.5, 0], [0.5, 0.5, 0]]], [0, 0, 1], 2],
            [[0, 0, 1], 150e6, "omnidirectional"],
            2,
        ),
        (
            [[[[-1.1, 0, 0], [0.9, 0, 0]]], [0, np.sqrt(2) / 2, np.sqrt(2) / 2]],
            [[0, np.sqrt(2) / 2, np.sqrt(2) / 2], 150e6],
            1,
        ),
        (
            [[[[1e8, -1, 0], [1e8, 1, 0]]], [0, 0, 1]],
            [[1 / np.sqrt(2), 0, 1 / np.sqrt(2)], 150e6, "omnidirectional"],
            (
                np.exp(1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
                + np.exp(-1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
            )
            / 2,
        ),
        (
            [[[[-1, -1, 0], [-1, 1, 0]], [[1, 1, 0]], [[1, -1, 0]]], [0, 0, 1], 4j],
            [[0, 0, 1], 150e6, "simplified"],
            4j,
        ),
    ],
)
def test_Station_response(args_init, args_response, expected):
    from beam_errors.array import Station

    test_station = Station(*args_init)
    npt.assert_almost_equal(
        test_station.calculate_response(*args_response), expected, 5
    )


@pytest.mark.parametrize(
    "args, expected",
    [
        ([0, 0, None], True),
        ([0, 321, 123], True),
        ([123, 0, None], True),
        ([1, 2, 5], True),
        ([1, 2, None], False),
    ],
)
def test_add_random_gain_drift(args, expected):
    from beam_errors.array import Station

    # Create array
    positions = [[[i, j, 0] for i in range(10)] for j in range(1000)]
    station = Station(positions)

    station.add_random_gain_drift(*args)

    tile_gains_std = np.std(station.get_element_property("g"))
    npt.assert_almost_equal(tile_gains_std, args[0], 0)

    antenna_gains = np.array(
        [tile.get_element_property("g") for tile in station.elements]
    )
    antenna_gains_std = np.std(antenna_gains)
    npt.assert_almost_equal(antenna_gains_std, args[1], 0)

    station.reset_elements()
    station.add_random_gain_drift(*args)
    new_antenna_gains = np.array(
        [tile.get_element_property("g") for tile in station.elements]
    )
    is_same = (antenna_gains == new_antenna_gains).all()
    npt.assert_equal(is_same, expected)
