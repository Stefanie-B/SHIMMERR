import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "path, number_of_stations, station_name, number_of_tiles, tile_number, number_of_dipoles, dipole_number, position",
    [
        ("files/arrays/single_baseline.txt", 2, "A", 1, 0, 1, 0, [1e8, 0, -1]),
        ("files/arrays/single_baseline.txt", 2, "B", 1, 0, 1, 0, [1e8, 0, 1]),
    ],
)
def test_load_array_from_file(
    path,
    number_of_stations,
    station_name,
    number_of_tiles,
    tile_number,
    number_of_dipoles,
    dipole_number,
    position,
):
    from beam_errors.load_array import load_array_from_file

    test_array = load_array_from_file(path)
    npt.assert_equal(len(test_array), number_of_stations)

    test_station = test_array[station_name]
    npt.assert_equal(len(test_station.elements), number_of_tiles)

    npt.assert_equal(
        len(test_station.elements[tile_number].elements), number_of_dipoles
    )

    npt.assert_array_equal(
        test_station.elements[tile_number].elements[dipole_number].p, position
    )


@pytest.mark.parametrize(
    "mode, number_of_stations, station_name, station_position, number_of_tiles, tile_number, number_of_dipoles",
    [
        ("CS", 48, "CS001HBA0", [3826896.63, 460979.131, 5064657.943], 24, 0, 16),
        (
            "Dutch_tapered",
            62,
            "CS001HBA0",
            [3826896.63, 460979.131, 5064657.943],
            24,
            5,
            16,
        ),
        (
            "Dutch_tapered",
            62,
            "RS509HBA",
            [3783537.922, 450129.744, 5097865.889],
            24,
            5,
            16,
        ),
        (
            "Dutch_sensitive",
            62,
            "CS001HBA0",
            [3826896.63, 460979.131, 5064657.943],
            24,
            10,
            16,
        ),
        (
            "Dutch_sensitive",
            62,
            "RS509HBA",
            [3783537.922, 450129.744, 5097865.889],
            48,
            10,
            16,
        ),
        (
            "international",
            76,
            "CS001HBA0",
            [3826896.63, 460979.131, 5064657.943],
            24,
            21,
            16,
        ),
        (
            "international",
            76,
            "RS509HBA",
            [3783537.922, 450129.744, 5097865.889],
            48,
            21,
            16,
        ),
        (
            "international",
            76,
            "UK608HBA",
            [4008462.280, -100376.948, 4943716.600],
            96,
            21,
            16,
        ),
        ("EoR", 56, "CS001HBA0", [3826896.63, 460979.131, 5064657.943], 24, 23, 16),
        ("EoR", 56, "RS106HBA", [3829205.994, 469142.209, 5062180.742], 24, 23, 16),
    ],
)
def test_load_LOFAR(
    mode,
    number_of_stations,
    station_name,
    station_position,
    number_of_tiles,
    tile_number,
    number_of_dipoles,
):
    from beam_errors.load_array import load_LOFAR

    test_array = load_LOFAR(mode=mode)
    npt.assert_equal(len(test_array), number_of_stations)

    test_station = test_array[station_name]
    npt.assert_array_almost_equal(
        test_station.p, station_position, 2
    )  # cm accuracy (as does lofar antpos)
    npt.assert_equal(len(test_station.elements), number_of_tiles)

    npt.assert_equal(
        len(test_station.elements[tile_number].elements), number_of_dipoles
    )
