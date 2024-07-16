import numpy as np
import numbers
from joblib import Parallel, delayed
from numba import jit, complex128, float64
import warnings

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
    parallel=True,
)
def calculate_array_factor_contribution(positions, gains, k, pointing, directions):
    """
    Calculates the array factor contributions of all elements and passes them separately (not summed).

    Parameters
    ----------
    positions : ndarray
        3xN array of element positions (must be floats)
    gains : ndarray
        N sized array of element gains (must be complex)
    k : float
        wave number
    pointing : ndarray
        3 sized array of pointing direction (unit length. must be float)
    directions : ndarray
        3xM array of directions to calculate the response in

    Returns
    -------
    ndarray
        N x M array with responses
    """
    positions = np.ascontiguousarray(positions)
    directions = np.ascontiguousarray(directions)
    gains = np.ascontiguousarray(gains)
    pointing = np.ascontiguousarray(pointing)
    relative_pointing = directions - pointing[:, np.newaxis]
    phase_delay = k * np.dot(positions, relative_pointing)
    return gains[:, np.newaxis] * np.exp(1j * phase_delay)


class Antenna:
    """
    A class that represents the lowest level elements in the array.

    Attributes
    -------------
    p: list
        Contains the 3D position of the antenna in ECEF (Earth-Centered Earth-Fixed) coordinates
    p_ENU: list
        Contains the 3D position of the antenna in ENU (East-North-Up) coordinates
    g: complex
        Complex gain of the element
    """

    def __init__(self, position, gain=1.0 + 0j):
        """
        Parameters
        ----------
        position : list
            Contains the 3D position of the antenna in ETRS coordinates
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
        Updates antenna gain
        """
        if not isinstance(new_gain, numbers.Complex):
            raise TypeError(f"Gain of {new_gain} not a permitted complex number.")
        self.g = complex(new_gain)

    def calculate_response(self, directions, frequency, mode="omnidirectional"):
        """
        Calculates the element response in directions.

        Parameters
        ----------
        directions : ndarray
            3xM array of unit length vectors that decribe the directions in ENU coordinates
        frequency : float
            Frequency of the measurement
        mode : str, optional
            sets the beam shape, by default "omnidirectional" (no directivity).

        Returns
        -------
        ndarray
            M length response
        """
        # Check if directions are fed correctly
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
        Contains the 3D position of the tile in ECEF (Earth-Centered Earth-Fixed) coordinates
    p_ENU: list
        Contains the 3D position of the tile in ENU (East-North-Up) coordinates
    g: complex
        Complex gain of the element
    d: ndarray
        Complex pointing of the tile (in ENU coordinates), where b is the geometric phase delay)
    """

    def __init__(self, positions, pointing, gain=1.0 + 0j):
        """
        Parameters
        ----------
        positions : list
            Contains the 3D position of the elements in ECEF coordinates
        pointing : complex, optional
            Complex pointing of the tile (in ENU coordinates)
        gain : complex, optional
            Complex gain of the tile (shared by all elements), by default 1
        """
        self.d = np.array(pointing).astype(float)
        self.g = complex(gain)

        # The gain of the tile is already applied, so the Antenna gain should be unity to avoid applying it twice
        self.elements = [Antenna(position) for position in positions]
        self.p = np.mean([element.p for element in self.elements], axis=0)

        self.p_ENU = None

    def reset_elements(self):
        """
        Resets all elements in the tile to have unit gain
        """
        [element.update_antenna(new_gain=1) for element in self.elements]

    def set_ENU_positions(self, rotation_matrix, station_position):
        """
        Calculates the ENU (East-North-Up) coordinates of all elements and the tile itself, based on the station rotation matrix from ECEF to ENU.
        """
        self.p_ENU = np.dot(rotation_matrix, self.p)
        for element in self.elements:
            element.p_ENU = np.dot(rotation_matrix, element.p) - self.p_ENU

        # We need to subtract the station position to make the station origin centered
        self.p_ENU -= np.dot(rotation_matrix, station_position)

    def get_element_property(self, property):
        """
        Helper function for the element list. Retrieves a property of the underlying elements.
        """
        return np.array([getattr(element, property) for element in self.elements])

    def set_element_property(self, property, values):
        [
            setattr(element, property, value)
            for element, value in zip(self.elements, values)
        ]

    def _break_number_of_elements(self, rng, n):
        """
        Randomly breaks a number of elements in the tile

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator for determining the elements to be broken
        n : int
            Number of elements to be broken
        """
        if n < 0:
            warnings.warn(
                "You are trying to break a negative number of elements in this tile. I am breaking none."
            )
            return
        if n == 0:
            return
        if n >= len(self.elements):
            warnings.warn(
                "You are trying to break all elements in the tile (or more). I am breaking all."
            )
            self.set_element_property("g", [0 for _ in self.elements])
            return

        # Find broken indices
        element_indices = list(range(len(self.elements)))
        rng.shuffle(element_indices)
        broken_elements = element_indices[:n]

        for i in broken_elements:
            self.elements[i].g = 0

    def calculate_response(self, directions, frequency, antenna_beams=None):
        """
        Calculates the tile response or array factor in M directions.

        Parameters
        ----------
        directions : ndarray
            3xM array of unit length vectors that decribe the directions in ENU coordinates
        frequency : float
            Frequency of the measurement
        antenna_beams : None or ndarray, optional
            If set to None (default) this disables the element beams and only the array factor is returned. Otherwise, give an N x M array of element responses in the requested directions

        Returns
        -------
        ndarray
            M length response
        """
        k = 2 * np.pi * frequency / c

        # Check if directions are given in the correct format. We explicitly cast to floats to work with jit later
        directions = np.array(directions, dtype=float).reshape(3, -1)
        if not np.allclose(
            np.linalg.norm(directions, axis=0), np.ones(directions.shape[1])
        ):
            raise ValueError("The directions are not unit length")

        # Calculate the gemetric response of the antenna elements (array factor)
        antenna_factors = calculate_array_factor_contribution(
            positions=self.get_element_property("p_ENU"),
            gains=self.get_element_property("g"),
            k=k,
            pointing=self.d,
            directions=directions,
        )

        # Sum over elements with or without beam
        if antenna_beams is None:
            tile_beam = np.mean(antenna_factors, axis=0)
        else:
            antenna_beams = np.array(antenna_beams)
            antenna_responses = antenna_beams * antenna_factors
            tile_beam = np.mean(antenna_responses, axis=0)

        return self.g * tile_beam


class Station:
    """
    A class that represents the full station.

    Attributes
    -------------
    p: list
        Contains the 3D position of the station in ECEF (Earth-Centered Earth-Fixed) coordinates
    p_ENU: list
        Contains the 3D position of the station in ENU (East-North-Up) coordinates
    g: complex
        Complex gain of the element
    d: ndarray
        Complex pointing of the station (in ENU coordinates), where b is the geometric phase delay)
    """

    def __init__(self, positions, pointing=[0, 0, 1], gain=1.0 + 0j):
        """
        Parameters
        ----------
        positions : list
            Contains the 3D position of the elements (tiles) in ECEF (Earth-Centered Earth-Fixed) coordinates
        pointing : complex, optional
            Complex pointing of the station in ENU (East-North-Up) coordinates, default is [0,0,1] (zenith)
        gain : complex, optional
            Complex gain of the tile (shared by all elements), by default 1
        """
        self.d = np.array(pointing).astype(float)
        self.g = complex(gain)

        # Set the tiles
        self.elements = [
            Tile(per_tile_positions, self.d) for per_tile_positions in positions
        ]
        self.p = np.mean([element.p for element in self.elements], axis=0)

        # Set the local coordinate frame (ENU)
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
                self.set_element_property(
                    "d", np.repeat(new_pointing, len(self.elements))
                )
                self.set_element_property(
                    "g", [element.g - self.g + new_gain for element in self.elements]
                )
                self.d = np.array(new_pointing).astype(float)
            else:
                self.set_element_property(
                    "g", [element.g - self.g + new_gain for element in self.elements]
                )
            self.g = complex(new_gain)
        else:
            self.set_element_property("d", np.repeat(new_pointing, len(self.elements)))
            self.d = np.array(new_pointing).astype(float)

    def reset_elements(self):
        """
        Resets all elements in the tile to the common pointing and unit gain
        """
        self.set_element_property("d", np.repeat(self.d, len(self.elements)))
        self.set_element_property("g", np.repeat(1, len(self.elements)))
        [tile.reset_elements() for tile in self.elements]

    @staticmethod
    def _draw_gaussian_complex_number(rng, sigma):
        """
        Draws a random ccomplex number from a normal distribution.
        -------
        _type_
            _description_
        """
        real, imag = rng.standard_normal(2)
        number = (
            (real + 1j * imag) * sigma / np.sqrt(2)
        )  # because we add 2 random numbers
        return number

    def add_random_gain_drift(self, sigma_tile, sigma_antenna, seed=None):
        """
        Add complex Gaussian zero mean noise to the gains.

        Parameters
        ----------
        sigma_tile : float
            Standard deviation of the noise added on a tile level
        sigma_antenna : float
            Standard deviation of the noise added on an antenna level
        seed : None or int, optional
            Seed of the random generator. Set by an integer for reproducability, by default None
        """
        rng = np.random.default_rng(seed=seed)
        for tile in self.elements:
            tile.g += self._draw_gaussian_complex_number(rng, sigma_tile)
            antenna_gains = [
                element.g + self._draw_gaussian_complex_number(rng, sigma_antenna)
                for element in tile.elements
            ]
            tile.set_element_property("g", antenna_gains)

    def add_spatial_random_gain_drift(self, sigma_tile, sigma_antenna, seed=None):
        pass

    def break_elements(self, mode="maximum", number=0, seed=None):
        """
        Breaks elements within the tiles of the array (by setting their gains to 0). Set the seed for reproducability, but be aware that a different seed should be used for different stations to guarantee randomness.

        Parameters
        ----------
        mode : str, optional
            Sets the way in which elements are broken, by default "maximum"
            maximum: Uniformly breaks elements up to a maximum number. If higher than the number of elements per tile, the full tile will be flagged more often (numbers between #elements and max are shifted to max)
            number: Breaks the specified number of elements. If higher than the number of antennas in a tile, all antennas are broken. Rounds to nearest number of elements per tile (so 10% of 16 elements = 1.6 elements --> 2 elements flagged in every tile)
            percentage: Same as number but with a percentage.
            typical: Breaks elements according to a normal(number, number) distribution, such that on average <number> elements are broken.
            typical_percentage: the same as above but with a percentage.
        number : int, optional
            Number that controls how many elements are broken (so maximum, percentage, etc.), by default 0
        seed : int, optional
            seed for the random number generator that controls how many and which elements are broken, by default None (initialize randomly every call)

        Raises
        ------
        ValueError
            either for a negative number of elements (not physically possible) or for an unknown mode
        """
        if number < 0:
            raise ValueError("number should be non-negative")
        rng = np.random.default_rng(seed=seed)
        if mode == "maximum":
            number_of_broken_elements = rng.integers(
                low=0, high=number + 1, size=len(self.elements)
            )
        elif mode == "number":
            number_of_broken_elements = [number for _ in self.elements]
        elif mode == "percentage":
            number_of_broken_elements = [
                int(np.rint(number / 100 * len(tile.elements)))
                for tile in self.elements
            ]
        elif mode == "typical":
            # Note that this may trigger warnings as the drawn random number can be outside the number of elements in a tile
            normal_distribution = rng.standard_normal(
                size=len(self.elements)
            )  # N(0, 1)
            number_of_broken_elements = np.rint(
                normal_distribution * np.sqrt(number) + number
            ).astype(
                int
            )  # N(number, number)
        elif mode == "typical_percentage":
            # Note that this may trigger warnings as the drawn random number can be outside the number of elements in a tile
            normal_distribution = rng.standard_normal(
                size=len(self.elements)
            )  # similar to above, but now we multiply with a percentage per tile
            number_of_broken_elements = [
                int(
                    np.rint(
                        x * np.sqrt(number * len(tile.elements) / 100)
                        + number * len(tile.elements) / 100
                    )
                )
                for x, tile in zip(normal_distribution, self.elements)
            ]
        else:
            raise ValueError("Other modes not yet implemented.")
        [
            tile._break_number_of_elements(rng, n)
            for n, tile in zip(number_of_broken_elements, self.elements)
        ]

    def ENU_rotation_matrix(self):
        """
        Calculates the 3x3 matrix that transforms the coordinates from ECEF (Earth-Centered Earth-Fixed) to ENU (East-North-Up)
        """
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
        """
        Sets ENU (East-North-Up) coordinates.
        """
        [
            tile.set_ENU_positions(self.ENU_rotation_matrix(), self.p)
            for tile in self.elements
        ]

    def get_element_property(self, property):
        """
        Helper function for the element list. Retrieves a property of the underlying elements.
        """
        return np.array([getattr(element, property) for element in self.elements])

    def set_element_property(self, property, values):
        [
            setattr(element, property, value)
            for element, value in zip(self.elements, values)
        ]

    def calculate_array_factor(self, directions, frequency, tile_beams=None):
        """
        Calculates the station array factor in M directions.

        Parameters
        ----------
        directions : ndarray
            3xM array of unit length vectors that decribe the directions in ENU coordinates
        frequency : float
            Frequency of the measurement
        tile_beams : None or ndarray, optional
            If set to None (default) this disables the element beams and only the array factor is returned. Otherwise, give an N x M array of element responses in the requested directions

        Returns
        -------
        ndarray
            M length response
        """
        k = 2 * np.pi * frequency / c

        # Make sure the directions are fed correctly. They must be floats for the jit array factor to work
        directions = np.array(directions, dtype=float).reshape(3, -1)
        if not np.allclose(
            np.linalg.norm(directions, axis=0), np.ones(directions.shape[1])
        ):
            raise ValueError("The directions are not unit length")

        # Calculate the gemetric response of the tile elements (array factor)
        tile_factors = calculate_array_factor_contribution(
            positions=self.get_element_property("p_ENU"),
            gains=self.get_element_property("g"),
            k=k,
            pointing=self.d,
            directions=directions,
        )

        # Sum over the tiles
        if tile_beams is None:
            station_beam = np.mean(tile_factors, axis=0)
        else:
            tile_beams = np.array(tile_beams)
            tile_responses = tile_beams * tile_factors
            station_beam = np.mean(tile_responses, axis=0)

        return self.g * station_beam

    def calculate_response(self, directions, frequency, antenna_mode=None):
        """
        Calculates the full station beam in M directions.

        Parameters
        ----------
        directions : ndarray
            3xM array of unit length vectors that decribe the directions in ENU coordinates
        frequency : float
            Frequency of the measurement
        antenna_mode : None or str, optional
            If set to None (default) this disables the element beams and only the array factor is returned. Otherwise, give the requested antenna type.

        Returns
        -------
        ndarray
            M length response
        """
        if antenna_mode is not None:
            # Calculate antenna beams (in parallel for large arrays)
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

            # Calculate the gemetric delays of the antennas to get the full tile beams
            tile_beams = [
                tile.calculate_response(
                    directions=directions,
                    frequency=frequency,
                    antenna_beams=antenna_beams[tile_number],
                )
                for tile_number, tile in enumerate(self.elements)
            ]

        else:
            # Array factor only option
            tile_beams = [
                tile.calculate_response(frequency=frequency, directions=directions)
                for tile in self.elements
            ]

        # Combine the tiles with geometric delay
        station_beam = self.calculate_array_factor(
            directions=directions,
            frequency=frequency,
            tile_beams=tile_beams,
        )

        return station_beam
