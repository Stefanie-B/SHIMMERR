import numpy as np
from joblib import Parallel, delayed

# For LOFAR specific
from lofarantpos.db import LofarAntennaDatabase
from lofarantpos import geo

c = 299792458  # m/s


class Antenna:
    """
    A class that represents the lowest level elements in the array. The element beam is not included, as it is common between
    elements and therefore only computed once in the highest level.

    Attributes
    -------------
    p: list
        Contains the 3D position of the antenna in ETRS coordinates
    g: complex
        Complex gain of the element
    s: complex
        Complex pointing of the element (in principle computed as exp(j b), where b is the geometric phase delay)
    w: complex
        Weight within the array factor for this element. This is given by the product of the gain and the pointing
    """

    def __init__(self, position, gain=1.0 + 0j):
        """
        Parameters
        ----------
        position : list
            Contains the 3D position of the antenna in ETRS coordinates
        pointing : complex, optional
            Complex pointing of the element (in principle computed as exp(j b), where b is the geometric phase delay), by default 1 (zenith)
        gain : complex, optional
            Complex gain of the element, by default 1
        """
        self.p = position
        self.g = gain

    def update_antenna(self, new_gain=None):
        """
        Updates antenna weight based on a new pointing vector or new gain
        """
        if new_gain is not None:
            self.g = new_gain

    def update_common_settings(self, new_g, old_g=0 + 0j):
        new_gain = self.g - old_g + new_g
        self.update_antenna(new_gain)

    def calculate_response(self, direction, frequency, mode="omnidirectional"):
        # TODO: implement Gaussian beam
        if mode == "omnidirectional":
            return self.g
        else:
            raise ValueError("Lowest level antenna mode {mode} not implemented.")


class Tile:
    """
    A class that represents the analogue beamformer elements in the array.

    Attributes
    -------------
    p: list
        Contains the 3D position of the tile, defined as the mean of all element positions
    g: complex
        Complex gain of the tile, shared between the elements (element gain = common gain + individual gain)
    s: complex
        Complex pointing of the tile (in principle computed as exp(j b), where b is the geometric phase delay)
    """

    def __init__(self, positions, pointing, gain=1.0 + 0j):
        """
        Parameters
        ----------
        positions : list
            Contains the 3D position of the elements in ETRS coordinates
        pointing : complex, optional
            Complex pointing of the tile (in principle computed as exp(j b), where b is the geometric phase delay), by default 1 (zenith)
        gain : complex, optional
            Complex gain of the tile (shared by all elements), by default 1
        """
        self.p = np.mean([element.p for element in self.elements], axis=0)
        self.d = pointing
        self.g = gain
        self.elements = [Antenna(position, self.g) for position in positions]

    def update_tile(self, new_pointing=None, new_gain=None):
        """
        Updates common tile gain or pointing.

        The new gains of the elements are set as:
        new_element_gain = old_element_gain - old_common_gain + new_common_gain
        This retains perturbations set on the elements
        """
        if new_gain is not None:
            [
                element.update_common_settings(new_g=new_gain, old_g=self.g)
                for element in self.elements
            ]
            self.g = new_gain
        if new_pointing is not None:
            self.d = new_pointing

    def reset_elements(self):
        """
        Resets all elements in the tile to the common gain and pointing (removing individual perturbations)
        """
        [element.update_antenna(new_gain=self.g) for element in self.elements]

    def calculate_response(self, direction, frequency, antenna_mode="omnidirectional"):
        k = 2 * np.pi * frequency / c

        def element_response_wrapper(element, direction, frequency, mode):
            antenna_beam = element.calculate_response(
                direction=direction, frequency=frequency, mode=mode
            )
            direction_offset_from_pointing = direction - self.d
            progessive_phase_delay = k * element.p @ direction_offset_from_pointing
            array_factor = np.exp(1j * progessive_phase_delay)
            return antenna_beam * array_factor

        element_responses = Parallel(n_jobs=-1)(
            delayed(element_response_wrapper)(
                element=element,
                direction=direction,
                frequency=frequency,
                mode=antenna_mode,
            )
            for element in self.elements
        )

        tile_response = sum(element_responses)
        return tile_response


class Station:
    def __init__(self, positions, pointing=1.0 + 0j, gain=1.0 + 0j):
        """
        Parameters
        ----------
        positions : list
            Contains the 3D position of the elements in ETRS coordinates
        pointing : complex, optional
            Complex pointing of the tile (in principle computed as exp(j b), where b is the geometric phase delay), by default 1 (zenith)
        gain : complex, optional
            Complex gain of the tile (shared by all elements), by default 1
        """
        self.d = pointing
        self.g = gain

        self.p = np.mean(positions, axis=(0, 1))

        self.elements = [
            Tile(per_tile_positions, self.d, self.g) for per_tile_positions in positions
        ]

    def update_station(self, new_pointing=None, new_gain=None):
        """
        Updates common tile gain. The new gains of the elements are set as:
        new_element_gain = old_element_gain - old_common_gain + new_common_gain
        This retains perturbations set on the elements
        """
        if new_gain is not None:
            [
                element.update_tile(
                    new_pointing=new_pointing, new_gain=element.g - self.g + new_gain
                )
                for element in self.elements
            ]
            self.g = new_gain
        else:
            [
                element.update_tile(new_pointing=new_pointing)
                for element in self.elements
            ]

    def reset_elements(self):
        """
        Resets all elements in the tile to the common gain and pointing (removing individual perturbations)
        """
        [
            element.update_tile(new_pointing=self.d, new_gain=self.g)
            for element in self.elements
        ]

    def calculate_response(self, direction, frequency, antenna_mode="omnidirectional"):
        k = 2 * np.pi * frequency / c

        def element_response_wrapper(element, direction, frequency, mode):
            tile_beam = element.calculate_response(
                direction=direction, frequency=frequency, antenna_mode=antenna_mode
            )
            direction_offset_from_pointing = direction - self.d
            progessive_phase_delay = k * element.p @ direction_offset_from_pointing
            array_factor = np.exp(1j * progessive_phase_delay)
            return tile_beam * array_factor

        element_responses = Parallel(n_jobs=-1)(
            delayed(element_response_wrapper)(
                element=element,
                direction=direction,
                frequency=frequency,
                mode=antenna_mode,
            )
            for element in self.elements
        )

        station_response = sum(element_responses)
        return station_response


def read_array_from_file(filepath, pointing=1 + 0j):
    # Stations separated by empty lines
    # Tiles are the first number
    # Next 3 characters are position etrs
    # final one is complex gain
    # in either lat-lon or in etrs, or provide helper functions
    # gains
    # add n_jobs param
    # check if jit helps

    array = []
    constructing_station = []
    tile = []
    with open(filepath, "r") as f:
        inputline = f.readline().strip("\n")
        if inputline == "":
            # station done
            full_station = Station(positions=constructing_station, pointing=pointing)
            array.append(full_station)

            constructing_station = []
            tile = []

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

    # Add the last station
    full_station = Station(positions=constructing_station, pointing=pointing)
    array.append(full_station)

    return array
