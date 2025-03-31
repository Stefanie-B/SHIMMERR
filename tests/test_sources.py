import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "args, expected",
    [
        ("2:33:06.8705", 38.2786270833),
        ("-8:05:45.6759", -121.44031625),
        ("19:59:26", 299.858333333),
        ("00:00:0", 0),
    ],
)
def test_parse_right_ascension(args, expected):
    from shimmerr.sources import Source

    npt.assert_almost_equal(Source.parse_right_ascension(args), expected)


@pytest.mark.parametrize(
    "args, expected",
    [
        ("90.00.00.00", 90),
        ("-40.44.00", -40.733333333),
        ("+58.48.42.4", 58.8117777778),
        ("87.34.13.324", 87.5703677778),
    ],
)
def test_parse_declination(args, expected):
    from shimmerr.sources import Source

    npt.assert_almost_equal(Source.parse_declination(args), expected)


@pytest.mark.parametrize(
    "args, expected",
    [
        ([150e6, 10, 150e6, "[0]"], 10),
        ([130e6, 20, 150e6, "[0]"], 20),
        ([130e6, 30, 150e6, "[-1]"], 34.6153846154),
        ([150e6, 40, 150e6, "[-1,2]"], 40),
        ([150e6, 27.477, 74e6, "[-0.158, 0.032, -0.180]"], 24.45105829),
    ],
)
def test_logarithmic_spectral_index_brightness(args, expected):
    from shimmerr.sources import Source

    npt.assert_almost_equal(
        Source.logarithmic_spectral_index_brightness(*args), expected
    )


@pytest.mark.parametrize(
    "args, expected",
    [
        ([150e6, 10, 150e6, "[0]"], 10),
        ([130e6, 20, 150e6, "[0]"], 20),
        ([150e6, 30, 130e6, "[-1]"], 29),
        ([150e6, 40, 150e6, "[-1,2]"], 39),
        ([74e6, 27.477, 150e6, "[-0.158, 0.032, -0.180]"], 27.2565787),
    ],
)
def test_linear_spectral_index_brightness(args, expected):
    from shimmerr.sources import Source

    npt.assert_almost_equal(Source.linear_spectral_index_brightness(*args), expected)


@pytest.mark.parametrize(
    "args_init, test_frequency, expected",
    [
        (
            ["2:33:06.8705", "90.00.00.00", 40, 150000000, "[-1,2]", True],
            150e6,
            [38.2786270833, 90, 40],
        ),
        (
            ["-8:05:45.6759", "-40.44.00", 40, 150e6, "[-1,2]", False],
            150e6,
            [-121.44031625, -40.733333333, 39],
        ),
        (
            [
                "19:59:26",
                "+58.48.42.4",
                27.477,
                150e6,
                "[-0.158, 0.032, -0.180]",
                False,
            ],
            74.0e6,
            [299.858333333, 58.8117777778, 27.2565787],
        ),
        (
            [
                "00:00:0",
                "87.34.13.324",
                27.477,
                74000000.0,
                "[-0.158, 0.032, -0.180]",
                True,
            ],
            150e6,
            [0, 87.5703677778, 24.45105829],
        ),
    ],
)
def test_Source_initialization(args_init, test_frequency, expected):
    from shimmerr.sources import Source

    test_source = Source(*args_init)
    npt.assert_almost_equal(test_source.ra, expected[0])
    npt.assert_almost_equal(test_source.dec, expected[1])
    npt.assert_almost_equal(test_source.I(test_frequency), expected[2])


@pytest.mark.parametrize(
    "args, expected",
    [(["19:59:26", "+58.48.42.4"], [299.858333333, 58.8117777778, {}])],
)
def test_Patch_initialization(args, expected):
    from shimmerr.sources import Patch

    test_patch = Patch(*args)

    npt.assert_almost_equal(test_patch.ra, expected[0])
    npt.assert_almost_equal(test_patch.dec, expected[1])
    npt.assert_equal(test_patch.elements.keys(), expected[2].keys())


@pytest.mark.parametrize(
    "args_patch, args_source1, args_source2, expected",
    [
        (
            ["19:59:26", "+58.48.42.4"],
            ["A", "2:33:06.8705", "90.00.00.00", 40, 150e6, "[-1,2]", True],
            ["B", "-8:05:45.6759", "-40.44.00", 40, 150e6, "[-1,2]", False],
            [38.2786270833, -40.733333333, 2],
        )
    ],
)
def test_add_source(args_patch, args_source1, args_source2, expected):
    from shimmerr.sources import Patch

    test_patch = Patch(*args_patch)
    test_patch.add_source(*args_source1)
    test_patch.add_source(*args_source2)
    npt.assert_almost_equal(test_patch.elements[args_source1[0]].ra, expected[0])
    npt.assert_almost_equal(test_patch.elements[args_source2[0]].dec, expected[1])
    npt.assert_equal(len(test_patch.elements.keys()), expected[2])


@pytest.mark.parametrize(
    "args, expected, expected_raises",
    [
        (
            ["files/skymodels/NCP_Cas_Cyg_3source.txt", "CygA", "sCygA", 150e6],
            [["NCP", "CasA", "CygA"], 40.733333333, 4999.2],
            None,
        ),
        (
            ["files/skymodels/NCP_Cas_Cyg_3source.txt", "sCygA", "sCygA", 150e6],
            [["NCP", "CasA", "CygA"], 40.733333333, 4998.37398374],
            KeyError,
        ),
        (
            ["tests/error_sky_model2.txt", "sCygA", "sCygA", 150e6],
            [["NCP", "CasA", "CygA"], 40.733333333, 4998.37398374],
            KeyError,
        ),
        (
            ["tests/error_sky_model.txt", "sCygA", "sCygA", 150e6],
            [["NCP", "CasA", "CygA"], 40.733333333, 4998.37398374],
            ValueError,
        ),
        (
            ["files/skymodels/NCP_simplified.txt", "Patch_0", "s0c1063", 150e6],
            [
                [
                    "3C61.1",
                    "Patch_0",
                    "Patch_1",
                    "Patch_10",
                    "Patch_11",
                    "Patch_12",
                    "Patch_13",
                    "Patch_14",
                    "Patch_15",
                    "Patch_16",
                    "Patch_17",
                    "Patch_18",
                    "Patch_19",
                    "Patch_2",
                    "Patch_20",
                    "Patch_21",
                    "Patch_22",
                    "Patch_3",
                    "Patch_4",
                    "Patch_5",
                    "Patch_6",
                    "Patch_7",
                    "Patch_8",
                    "Patch_9",
                ],
                89.4634721389,
                0.0331093630744605,
            ],
            None,
        ),
    ],
)
def test_Skymodel_init(args, expected, expected_raises):
    from shimmerr.sources import Skymodel

    if expected_raises is not None:
        with pytest.raises(expected_raises):
            test_skymodel = Skymodel(args[0])
            test_skymodel.elements[args[1]]
            test_skymodel.elements[args[1]].elements[args[2]]

    else:
        test_skymodel = Skymodel(args[0])
        npt.assert_equal(
            sorted(list(test_skymodel.elements.keys())), sorted(expected[0])
        )
        npt.assert_almost_equal(test_skymodel.elements[args[1]].dec, expected[1])
        npt.assert_almost_equal(
            test_skymodel.elements[args[1]].elements[args[2]].I(args[3]), expected[2]
        )
