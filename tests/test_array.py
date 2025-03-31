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
    from shimmerr.array import Antenna

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
        ([[0, 0, 0], 5j], [[0, 0, 1], 150e6], 1),
        ([[0, 0, 0], 5j], [[0, 1, 0], 150e6], 1),
        (
            [[0, 0, 0], 5j],
            [[0, 1 / np.sqrt(2), 1 / np.sqrt(2)], 150e6, "simplified"],
            np.exp(-2 * (np.pi / 4) ** 3),
        ),
    ],
)
def test_Antenna_response(args_init, args_response, expected):
    from shimmerr.array import Antenna

    test_antenna = Antenna(*args_init)
    npt.assert_almost_equal(test_antenna.calculate_response(*args_response), expected)


@pytest.mark.parametrize(
    "args, expected, expected_raises",
    [
        (
            [[[0, 0, 0], [1, 1, 1]], False, 2],
            [[0.5, 0.5, 0.5], False, 2, [0, 0, 0], 1],
            None,
        ),
        (
            [[[0, 4j, 0], [1, 1, 1]], [0, 0, 1], 2],
            [[0.5, 0.5, 0.5], [0, 0, 1], 2, [0, 0, 0], 1],
            TypeError,
        ),
        (
            [[[0, 0, 0], [1, 1, 1], [-7, -10, -1]], False, 2],
            [[-2, -3, 0], False, 2, [0, 0, 0], 1],
            None,
        ),
        (
            [[[0, 0, 0], [1, 1, 1], [-7, -10, -1]], True],
            [[-2, -3, 0], True, 1, [0, 0, 0], 1 + 0j],
            None,
        ),
        (
            [[[0, 0, 0, 0], [1, 1, 1], [-7, -10, -1]], False],
            [[-2, -3, 0], [0, 0, 1], 1, [0, 0, 0], 1],
            ValueError,
        ),
    ],
)
def test_Tile_init(args, expected, expected_raises):
    from shimmerr.array import Tile

    if expected_raises is not None:
        with pytest.raises(expected_raises):
            test_tile = Tile(*args)
    else:
        test_tile = Tile(*args)
        npt.assert_almost_equal(test_tile.p, expected[0], 5)
        npt.assert_equal(test_tile.tracking, expected[1])
        npt.assert_equal(test_tile.g, expected[2])
        npt.assert_equal(test_tile.elements[0].p, expected[3])
        npt.assert_array_almost_equal(test_tile.elements[0].g, expected[4], 5)


@pytest.mark.parametrize(
    "args_init, args_response, expected",
    [
        (
            [[[-1, -1, 0], [1, 1, 0]], False, 2],
            [[0, 0, 1], 150e6, None],
            1,
        ),
        (
            [[[-1, 0, 0], [1, 0, 0]], False],
            [[0, np.sqrt(2) / 2, np.sqrt(2) / 2], 150e6, None, [0, 0, 1]],
            1,
        ),
        (
            [[[-1, 0, 0], [1, 0, 0]], True],
            [[np.sqrt(2) / 2, 0, np.sqrt(2) / 2], 150e6, 1, [0, 0, 1]],
            (
                np.exp(1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
                + np.exp(-1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
            )
            / 2,
        ),
        (
            [[[-1, 0, 0], [1, 0, 0]], True],
            [[0, 0, 1], 150e6, 2, [0, np.sqrt(2) / 2, np.sqrt(2) / 2]],
            2,
        ),
        (
            [[[-1, 0, 0], [1, 0, 0]], False],
            [[np.sqrt(2) / 2, 0, np.sqrt(2) / 2], 150e6, [2, 2]],
            (
                2 * np.exp(1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
                + 2 * np.exp(-1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
            )
            / 2,
        ),
    ],
)
def test_Tile_response(args_init, args_response, expected):
    from shimmerr.array import Tile

    test_tile = Tile(*args_init)
    test_tile.set_ENU_positions(
        np.eye(3), np.array([0, 0, 0])
    )  # Since the tile is not part of a station it doesn't have an ENU position yet
    npt.assert_almost_equal(test_tile.calculate_response(*args_response), expected, 5)


@pytest.mark.parametrize(
    "args, expected, expected_raises",
    [
        (
            [[[[0, 0, 0], [1, 1, 1]]], None, None, 2],
            [[0.5, 0.5, 0.5], None, 2, [0.5, 0.5, 0.5], 1, [0, 0, 0], 1],
            None,
        ),
        (
            [[[[0, 4j, 0], [1, 1, 1]]], None, None, 2],
            [],
            TypeError,
        ),
        (
            [[[[0, 0, 0], [1, 1, 1]], [[-6.5, -10.5, -1.5]]], None, None, 2],
            [[-3, -5, -0.5], None, 2, [0.5, 0.5, 0.5], 1, [0, 0, 0], 1],
            None,
        ),
        (
            [[[[0, 0, 0]], [[1, 1, 1], [-7, -10, -1]]], 250, 80],
            [[-1.5, -2.25, 0], [250, 80], 1, [0, 0, 0], 1 + 0j, [0, 0, 0], 1 + 0j],
            None,
        ),
        (
            [[[[0, 0, 0]], [[1, 1, 1], [-7, -10, -1]]], None, -90],
            [[-1.5, -2.25, 0], [0, -90], 1, [0, 0, 0], 1 + 0j, [0, 0, 0], 1 + 0j],
            None,
        ),
        (
            [[[[0, 0, 0]], [[1, 1, 1], [-7, -10, -1]]], None, 80],
            [],
            ValueError,
        ),
        (
            [[[[0, 0, 0]], [[1, 1, 1], [-7, -10, -1]]], 250, 80],
            [[-1.5, -2.25, 0], [250, 80], 1, [0, 0, 0], 1 + 0j, [0, 0, 0], 1 + 0j],
            None,
        ),
        (
            [[[[0, 0, 0, 0], [1, 1, 1], [-7, -10, -1]]], None, None],
            [],
            ValueError,
        ),
        (
            [[[[0, 0, 0], [1, 1, 1]]], None, None, None],
            [],
            TypeError,
        ),
        (
            [[[[0, 0, 0], [1, 1, 1]]], 4j],
            [],
            TypeError,
        ),
    ],
)
def test_Station_init(args, expected, expected_raises):
    from shimmerr.array import Station

    if expected_raises is not None:
        with pytest.raises(expected_raises):
            test_station = Station(*args)
    else:
        test_station = Station(*args)
        npt.assert_almost_equal(test_station.p, expected[0], 5)
        if expected[1] is not None:
            npt.assert_equal(test_station.d["ra"], expected[1][0])
            npt.assert_equal(test_station.d["dec"], expected[1][1])
            npt.assert_equal(test_station.tracking, True)
        else:
            npt.assert_equal(test_station.d, expected[1])
            npt.assert_equal(test_station.tracking, False)
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
            [[[[-1.5, -1.5, 1e8], [0.5, 0.5, 1e8]]], None, None, 2],
            [[0, 0, 1], 150e6, "omnidirectional"],
            2,
        ),
        (
            [[[[1e8, 1e8, -1], [1e8, 1e8, 1]]], None, None, 2],
            [[0, 1 / np.sqrt(2), 1 / np.sqrt(2)], 150e6],
            (
                np.exp(1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
                + np.exp(-1j * np.sqrt(2) * np.pi * 150e6 / 299792458)
            ),
        ),
        (
            [[[[-1.5, -1.5, 1e8], [0.5, 0.5, 1e8]]], None, None, 2],
            [[1 / np.sqrt(2), 0, 1 / np.sqrt(2)], 150e6],
            2,
        ),
        (
            [
                [
                    [[-1, -1, 1e8], [-1, 1, 1e8]],
                    [[1, 1, 1e8]],
                    [[1, -1, 1e8]],
                ],
                None,
                None,
                4j,
            ],
            [[0, 0, 1], 150e6, "simplified"],
            4j,
        ),
    ],
)
def test_Station_response(args_init, args_response, expected):
    from shimmerr.array import Station

    test_station = Station(*args_init)
    npt.assert_almost_equal(
        test_station.calculate_response(*args_response), expected, 5
    )


@pytest.mark.parametrize(
    "args_init, args_location, expected",
    [
        (  # Vega, Indian Ocean
            [[[[2148527, 5903030, -1100249]]]],
            ["2024-07-09T22:42:33", 279.44875, 38.8064444444],
            [[-0.6715], [0.6821], [0.281194]],
        ),
        (  # Vega, Hidden Lake Territorial Park
            [[[[-1184057, -2700745, 5636532]]]],
            ["2024-01-10T10:59:00", 279.44875, 38.8064444444],
            [[0.6821], [0.6229], [0.382993]],
        ),
        (  # Alpheratz, South Africa
            [[[[5069019, 2362822, -3056109]]]],
            ["2024-12-05T17:59:00", 2.42083333333, 29.2311388889],
            [[-0.1114], [0.8449], [0.523269]],
        ),
        (  # Alpheratz, South Africa
            [[[[5069019, 2362822, -3056109]]]],
            ["2024-12-05T15:59:00", 2.42083333333, 29.2311388889, 7200, 2],
            [[0.3, -0.1114], [0.8, 0.8449], [0.5, 0.523269]],
        ),
        (  # Alpheratz, South Africa
            [[[[5069019, 2362822, -3056109]]]],
            ["2024-12-05T18:59:00", 2.42083333333, 29.2311388889, -3600, 2],
            [[-0.3, -0.1114], [0.8, 0.8449], [0.5, 0.523269]],
        ),
        (  # Alpheratz, South Africa, 360 az offset
            [[[[5069019, 2362822, -3056109]]]],
            ["2024-12-05T18:59:00", 362.42083333333, 29.2311388889, -3600, 2],
            [[-0.3, -0.1114], [0.8, 0.8449], [0.5, 0.523269]],
        ),
        (  # Shedar down, South Africa
            [[[[5069019, 2362822, -3056109]]]],
            ["2024-10-10T03:43:00", 10.4870833333, 56.6749166667],
            [[np.nan], [np.nan], [np.nan]],
        ),
    ],
)
def test_radec_to_ENU(args_init, args_location, expected):
    from shimmerr.array import Station

    test_station = Station(*args_init)
    npt.assert_array_almost_equal(
        test_station.radec_to_ENU(*args_location), expected, 1
    )


@pytest.mark.parametrize(
    "station_args,rotation_args,expected",
    [
        ([[[[1e8, 1e8, 1e8]]]], np.array([1, 1, 1]), np.array([0, 0, np.sqrt(3)])),
        ([[[[1e8, 1e8, 0]]]], np.array([0, 0, 1]), np.array([0, 1, 0])),
        ([[[[1e11, 0, 1e11]]]], np.array([0, 1, 0]), np.array([1, 0, 0])),
        (
            [[[[1e8, 0, 1e8]]]],
            np.array([0, 1, 1]),
            np.array([1, 1 / np.sqrt(2), 1 / np.sqrt(2)]),
        ),
        (
            [[[[4198606, 171151, 4782378]]]],
            np.array([339, 3596, -491]),
            np.array([3579.21, -688.57, -48.59]),
        ),
    ],
)
def test_ENU_rotation(station_args, rotation_args, expected):
    from shimmerr.array import Station

    test_station = Station(*station_args)
    rotation_matrix = test_station.ENU_rotation_matrix()

    npt.assert_array_almost_equal(rotation_matrix @ rotation_args, expected, 2)


@pytest.mark.parametrize(
    "args, expected",
    [
        ([0, 0, None], True),
        ([0, 321, 123], True),
        ([12, 0, None], True),
        ([1, 2, 5], True),
        ([1, 2, None], False),
    ],
)
def test_add_random_gain_drift(args, expected):
    from shimmerr.array import Station

    # Create array
    positions = [[[i, j, 0] for i in range(100)] for j in range(1000)]
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
