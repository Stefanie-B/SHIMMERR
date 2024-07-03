import numpy as np
import numbers
from joblib import Parallel, delayed
from numba import jit, complex128, float64

c = 299792458  # m/s


@jit(
    complex128[:, :](
        float64[:, :],
        complex128[:],
        float64,
        float64[:],
        float64[:, :],
    ),
    fastmath=True,
    nopython=True,
)
def calculate_array_factor_contribution(positions, gains, k, pointing, directions):
    positions = np.ascontiguousarray(positions)
    directions = np.ascontiguousarray(directions)
    gains = np.ascontiguousarray(gains)
    pointing = np.ascontiguousarray(pointing)
    relative_pointing = directions - pointing[:, np.newaxis]
    phase_delay = k * (positions @ relative_pointing)
    return gains[:, np.newaxis] * np.exp(1j * phase_delay)


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
        if not isinstance(gain, numbers.Complex):
            raise TypeError(f"Gain of {gain} not a permitted complex number.")

        position = np.array(position)
        if not len(position) == 3:
            raise ValueError(f"Element position {position} not 3 dimensional.")

        for p in range(3):
            if not isinstance(position[p], numbers.Real):
                raise TypeError(
                    f"Position element {p} equals {position[p]}, this is not a valid number."
                )

        self.p = position.astype(float)
        self.g = complex(gain)
        self.p_ENU = None

    def update_antenna(self, new_gain=None):
        """
        Updates antenna weight based on a new pointing vector or new gain
        """
        if not isinstance(new_gain, numbers.Complex):
            raise TypeError(f"Gain of {new_gain} not a permitted complex number.")
        if new_gain is not None:
            self.g = complex(new_gain)

    def update_common_settings(self, new_g, old_g=0 + 0j):
        new_gain = self.g - old_g + new_g
        self.update_antenna(new_gain)

    def calculate_response(self, directions, frequency, mode="omnidirectional"):
        directions = np.array(directions).reshape(3, -1)
        if not np.allclose(
            np.linalg.norm(directions, axis=0), np.ones(directions.shape[1])
        ):
            raise ValueError("The directions are not unit length")

        if mode == "omnidirectional":
            # Simply the gain, there is no directivity
            antenna_beam = np.array([self.g for _ in range(directions.shape[1])])
            return antenna_beam
        elif mode == "simplified":
            # USE WITH EXTREME CAUTION
            # This beam is more of an 'artist impression' of what a beam looks like than a serious simulation
            # you can use it to get a general feel for the effect of the element beam but not for quantative results
            inclinations = np.arccos(directions[2, :])  # assumes unit length direction
            beam_shape = np.exp(-(inclinations**3) * 2)
            return self.g * beam_shape
        else:
            raise ValueError(f"Lowest level antenna mode {mode} not implemented.")


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
        self.d = np.array(pointing).astype(float)
        self.g = complex(gain)

        # The gain of the tile is already applied, so the Antenna gain should be unity to avoid applying it twice
        self.elements = [Antenna(position) for position in positions]
        self.p = np.mean([element.p for element in self.elements], axis=0)

        self.p_ENU = None

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
            self.g = complex(new_gain)
        if new_pointing is not None:
            self.d = np.array(new_pointing).astype(float)

    def reset_elements(self):
        """
        Resets all elements in the tile to the common gain and pointing (removing individual perturbations)
        """
        [element.update_antenna(new_gain=1) for element in self.elements]

    def set_ENU_positions(self, rotation_matrix, station_position):
        self.p_ENU = np.dot(rotation_matrix, self.p)
        for element in self.elements:
            element.p_ENU = np.dot(rotation_matrix, element.p) - self.p_ENU
        self.p_ENU -= np.dot(rotation_matrix, station_position)

    def get_element_property(self, property):
        return np.array([getattr(element, property) for element in self.elements])

    def calculate_response(self, directions, frequency, antenna_beams=None):
        k = 2 * np.pi * frequency / c

        directions = np.array(directions, dtype=float).reshape(3, -1)
        if not np.allclose(
            np.linalg.norm(directions, axis=0), np.ones(directions.shape[1])
        ):
            raise ValueError("The directions are not unit length")
        antenna_factors = calculate_array_factor_contribution(
            positions=self.get_element_property("p_ENU"),
            gains=self.get_element_property("g"),
            k=k,
            pointing=self.d,
            directions=directions,
        )

        if antenna_beams is None:
            tile_beam = np.mean(antenna_factors, axis=0)
        else:
            antenna_beams = np.array(antenna_beams)
            antenna_responses = antenna_beams * antenna_factors
            tile_beam = np.mean(antenna_responses, axis=0)

        return self.g * tile_beam


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
        self.d = np.array(pointing).astype(float)
        self.g = complex(gain)

        # self.p = np.mean(np.array(positions), axis=(0, 1))

        self.elements = [
            Tile(per_tile_positions, self.d) for per_tile_positions in positions
        ]
        self.p = np.mean([element.p for element in self.elements], axis=0)
        if self.p[0] == 0 and self.p[1] == 0:
            raise ValueError(
                "Arrays pointing straight to the Earth's North pole are not implemented."
            )

        self.set_ENU_positions()

    def update_station(self, new_pointing=None, new_gain=None):
        """
        Updates common tile gain. The new gains of the elements are set as:
        new_element_gain = old_element_gain - old_common_gain + new_common_gain
        This retains perturbations set on the elements
        """
        if new_gain is not None:
            if new_pointing is not None:
                [
                    element.update_tile(
                        new_pointing=new_pointing,
                        new_gain=element.g - self.g + new_gain,
                    )
                    for element in self.elements
                ]
                self.d = np.array(new_pointing).astype(float)
            else:
                [
                    element.update_tile(new_gain=element.g - self.g + new_gain)
                    for element in self.elements
                ]
            self.g = complex(new_gain)
        else:
            [
                element.update_tile(new_pointing=new_pointing)
                for element in self.elements
            ]
            self.d = np.array(new_pointing).astype(float)

    def reset_elements(self):
        """
        Resets all elements in the tile to the common gain and pointing (removing individual perturbations)
        """
        [
            element.update_tile(new_pointing=self.d, new_gain=self.g)
            for element in self.elements
        ]

    def ENU_rotation_matrix(self):
        normal_vector = self.p / np.linalg.norm(self.p)

        # We calculate local North as the projection of the ECEF north vector on the space orthogonal to
        # the local normal vector
        true_north = [0, 0, 1]
        local_north = true_north - np.dot(true_north, normal_vector) * normal_vector
        local_north /= np.linalg.norm(local_north)

        # Local east is orthogonal to both the normal vector and local North
        local_east = np.cross(normal_vector, local_north)

        return np.array([local_east, local_north, normal_vector])

    def set_ENU_positions(self):
        [
            tile.set_ENU_positions(self.ENU_rotation_matrix(), self.p)
            for tile in self.elements
        ]

    def get_element_property(self, property):
        return np.array([getattr(element, property) for element in self.elements])

    def calculate_array_factor(self, directions, frequency, tile_beams=None):
        k = 2 * np.pi * frequency / c

        directions = np.array(directions, dtype=float).reshape(3, -1)
        if not np.allclose(
            np.linalg.norm(directions, axis=0), np.ones(directions.shape[1])
        ):
            raise ValueError("The directions are not unit length")

        tile_factors = calculate_array_factor_contribution(
            positions=self.get_element_property("p_ENU"),
            gains=self.get_element_property("g"),
            k=k,
            pointing=self.d,
            directions=directions,
        )

        if tile_beams is None:
            station_beam = np.mean(tile_factors, axis=0)
        else:
            tile_beams = np.array(tile_beams)
            tile_responses = tile_beams * tile_factors
            station_beam = np.mean(tile_responses, axis=0)

        return self.g * station_beam

    def calculate_response(self, directions, frequency, antenna_mode=None):

        if antenna_mode is not None:

            def element_response_wrapper(tile, frequency, directions, antenna_mode):
                antenna_beams = [
                    antenna.calculate_response(
                        frequency=frequency,
                        directions=directions,
                        mode=antenna_mode,
                    )
                    for antenna in tile.elements
                ]
                return antenna_beams

            antenna_beams = Parallel(n_jobs=-1)(
                delayed(element_response_wrapper)(
                    tile=tile,
                    directions=directions,
                    frequency=frequency,
                    antenna_mode=antenna_mode,
                )
                for tile in self.elements
            )

            tile_beams = [
                tile.calculate_response(
                    directions=directions,
                    frequency=frequency,
                    antenna_beams=antenna_beams[tile_number],
                )
                for tile_number, tile in enumerate(self.elements)
            ]
        else:
            tile_beams = [
                tile.calculate_response(frequency=frequency, directions=directions)
                for tile in self.elements
            ]
        station_beam = self.calculate_array_factor(
            directions=directions,
            frequency=frequency,
            tile_beams=tile_beams,
        )

        return station_beam
