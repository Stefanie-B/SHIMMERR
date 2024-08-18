from beam_errors.array import Station
import numpy as np

# For LOFAR specific
from lofarantpos.db import LofarAntennaDatabase


def load_array_from_file(filepath, pointing_ra=None, pointing_dec=None):
    array = {}
    constructing_station = []
    tile = []
    this_tile = None
    station_name = None
    with open(filepath, "r") as f:
        for line in f:
            inputline = line.strip("\n")
            if inputline.startswith("#"):
                continue
            elif inputline == "":
                if len(tile) > 0:
                    constructing_station.append(tile)
                if len(constructing_station) > 0:
                    # station done
                    full_station = Station(
                        positions=constructing_station,
                        pointing_ra=pointing_ra,
                        pointing_dec=pointing_dec,
                    )
                    array[station_name] = full_station

                    constructing_station = []
                    tile = []
                    station_name = None
            elif station_name is None:
                station_name = inputline
            else:
                tile_identifier, x, y, z = inputline.split(",")

                # Check if a new tile has started
                if tile != []:
                    if tile_identifier != this_tile:
                        constructing_station.append(tile)
                this_tile = tile_identifier

                # Add the element
                new_position = np.array([x, y, z]).astype(float)
                tile.append(new_position)

    if len(tile) > 0:
        constructing_station.append(tile)
    if len(constructing_station) > 0:
        # station done
        full_station = Station(
            positions=constructing_station,
            pointing_ra=pointing_ra,
            pointing_dec=pointing_dec,
        )
        array[station_name] = full_station

    return array


def load_LOFAR(mode="EoR", pointing_ra=None, pointing_dec=None):
    # Dutch, CS, international, EoR (gebruikte NL stations)
    db = LofarAntennaDatabase()
    antpos_stations = sorted(list(db.phase_centres.keys()))

    if mode == "CS":
        station_names = [
            station
            for station in antpos_stations
            if (station.endswith("HBA0") or station.endswith("HBA1"))
            and station.startswith("CS")
        ]
        taper_RS = False
    elif mode == "Dutch_tapered":
        station_names = [
            station
            for station in antpos_stations
            if (
                (station.endswith("HBA0") or station.endswith("HBA1"))
                and station.startswith("CS")
            )
            or (station.endswith("HBA") and station.startswith("RS"))
        ]
        taper_RS = True
    elif mode == "Dutch_sensitive":
        station_names = [
            station
            for station in antpos_stations
            if (
                (station.endswith("HBA0") or station.endswith("HBA1"))
                and station.startswith("CS")
            )
            or (station.endswith("HBA") and station.startswith("RS"))
        ]
        taper_RS = False
    elif mode == "international":
        station_names = [
            station
            for station in antpos_stations
            if (
                (station.endswith("HBA0") or station.endswith("HBA1"))
                and station.startswith("CS")
            )
            or (station.endswith("HBA") and not station.startswith("CS"))
        ]
        taper_RS = False
    elif mode == "EoR":
        station_names = [
            station
            for station in antpos_stations
            if (
                (station.endswith("HBA0") or station.endswith("HBA1"))
                and station.startswith("CS")
            )
            or (station.endswith("HBA") and station.startswith("RS"))
            and (
                station
                not in [
                    "RS208HBA",
                    "RS210HBA",
                    "RS310HBA",
                    "RS409HBA",
                    "RS508HBA",
                    "RS509HBA",
                ]
            )
        ]
        taper_RS = True
    else:
        raise ValueError(f"LOFAR mode {mode} not implemented.")

    taper_tiles = [
        0,
        1,
        2,
        3,
        4,
        7,
        8,
        9,
        14,
        15,
        16,
        23,
        24,
        31,
        32,
        33,
        38,
        39,
        40,
        43,
        44,
        45,
        46,
        47,
    ]

    array = {}
    for station_name in station_names:

        dipole_elements_pqr = db.hba_dipole_pqr(station_name)

        # Taper the RS by removing unused elements. We taper the tiles, so we have to divide the dipole number by 16 to get the tile number
        if taper_RS and station_name.startswith("RS"):
            filtered_dipole_elements_pqr = [
                dipole
                for dipole_number, dipole in enumerate(dipole_elements_pqr)
                if dipole_number // 16 not in taper_tiles
            ]
        else:
            filtered_dipole_elements_pqr = dipole_elements_pqr

        # rotate the dipoles from the local frame to ETRS
        rotation_matrix = db.pqr_to_etrs[station_name]
        station_center = db.phase_centres[station_name]
        dipole_elements_etrs = (
            filtered_dipole_elements_pqr @ rotation_matrix.T + station_center
        )

        # Split the tiles
        station_split_in_tiles = dipole_elements_etrs.reshape(-1, 16, 3)

        # create and add station object
        station = Station(
            station_split_in_tiles, pointing_ra=pointing_ra, pointing_dec=pointing_dec
        )
        array[station_name] = station

    return array
