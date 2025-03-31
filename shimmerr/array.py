import numpy as np
import numbers
import copy
from numba import jit, complex128, float64, prange
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
from astropy import constants as const
import warnings


@jit(
    complex128[:, :](
        float64[:, :],
        complex128[:],
        float64,
        float64[:, :],
    ),
    fastmath=True,
    nopython=True,
    parallel=True,
)
def calculate_array_factor_contribution(positions, gains, k, directions):
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

    phase_delay = k * np.dot(positions, directions)

    array_factor_contribution = np.empty(phase_delay.shape, dtype=np.complex128)
    for i in prange(phase_delay.shape[0]):
        array_factor_contribution[i, :] = gains[i] * np.exp(1j * phase_delay[i, :])
    return array_factor_contribution


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

        if mode == "omnidirectional":
            # Simply the gain, there is no directivity
            antenna_beam = np.array([1 for _ in range(directions.shape[1])])
            return antenna_beam
        elif mode == "simplified":
            # USE WITH EXTREME CAUTION
            # This beam is more of an 'artist impression' of what a beam looks like than a serious simulation
            # you can use it to get a general feel for the effect of the element beam but not for quantative results
            inclinations = np.arccos(directions[2, :])  # assumes unit length direction
            beam_shape = np.exp(-(inclinations**3) * 2)
            return beam_shape
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
        Complex pointing of the tile (in Ra, Dec)
    """

    def __init__(self, positions, tracking, gain=1.0 + 0j):
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
        self.tracking = tracking
        self.g = complex(gain)

        # The gain of the tile is already applied, so the Antenna gain should be unity to avoid applying it twice
        self.elements = [Antenna(position) for position in positions]
        self.p = np.mean([element.p for element in self.elements], axis=0)

        self.p_ENU = None

    def update_tile_gain(self, new_gain):
        [
            element.update_common_settings(new_g=new_gain, old_g=self.g)
            for element in self.elements
        ]
        self.g = complex(new_gain)

    def reset_elements(self):
        """
        Resets all elements in the tile to have unit gain
        """
        [element.update_antenna(new_gain=1.0 + 0j) for element in self.elements]

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

    def set_element_property(self, property, values, same_value=False):
        """
        Helper function for the element list. Sets a property of the underlying elements.
        """
        if same_value:
            [setattr(element, property, values) for element in self.elements]
        else:
            [
                setattr(element, property, value)
                for element, value in zip(self.elements, values)
            ]

    def calculate_response(
        self, directions, frequency, antenna_beams=None, pointing_directions=None
    ):
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
        pointing_directions : ndarray, optional
             3xM array of unit length vectors that decribe the direction of the pointing center in ENU coordinates. Default is None (drift-scan)

         Returns
         -------
         ndarray
             M length response
        """
        k = 2 * np.pi * frequency / const.c.value

        # Check if directions are given in the correct format. We explicitly cast to floats to work with jit later
        directions = np.array(directions, dtype=float).reshape(3, -1)

        # Adjust for the phase center
        if self.tracking:
            pointing_directions = np.array(pointing_directions, dtype=float).reshape(
                3, -1
            )
            directions -= pointing_directions

        # Calculate the gemetric response of the antenna elements (array factor)
        antenna_factors = calculate_array_factor_contribution(
            positions=self.get_element_property("p_ENU"),
            gains=self.get_element_property("g"),
            k=k,
            directions=directions,
        )

        # Sum over elements with or without beam
        if antenna_beams is None:
            tile_beam = np.mean(antenna_factors, axis=0)
        else:
            antenna_beams = np.array(antenna_beams)
            antenna_responses = antenna_beams * antenna_factors
            tile_beam = np.mean(antenna_responses, axis=0)

        return tile_beam


class Station:
    """
    A class that represents the full station.

    Attributes
    -------------
    p: list
        Contains the 3D position of the station in ECEF (Earth-Centered Earth-Fixed) coordinates
    p_array: list
        Contains the 3D position of the station in ENU (East-North-Up) coordinates of a specified reference stataion
    g: complex
        Complex gain of the element
    d: ndarray
        Complex pointing of the station (in Ra, Dec)
    tracking: bool
        If set, the telescope is in tracking mode (pointed to a specific point), if False, the telescope is in drift-scan mode
    """

    def __init__(self, positions, pointing_ra=None, pointing_dec=None, gain=1.0 + 0j):
        """
        Parameters
        ----------
        positions : list
            Contains the 3D position of the elements (tiles) in ECEF (Earth-Centered Earth-Fixed) coordinates
        pointing_ra : float, optional
            Right ascension of the phase center in deg, default is None
        pointing_dec : float, optional
            Declination of the phase center in deg, default is None
        gain : complex, optional
            Complex gain of the tile (shared by all elements), by default 1
        """
        self.elements = []
        self._set_pointing_center(pointing_ra, pointing_dec)
        self.g = complex(gain)

        # Set the tiles
        self.elements = [
            Tile(per_tile_positions, self.tracking) for per_tile_positions in positions
        ]
        self.p = np.mean([element.p for element in self.elements], axis=0)

        # Set the local coordinate frame (ENU)
        if self.p[0] == 0 and self.p[1] == 0:
            raise ValueError(
                "Arrays pointing straight to the Earth's North pole are not implemented."
            )
        self.set_ENU_positions()

    def _set_pointing_center(self, pointing_ra=None, pointing_dec=None):
        """
        Sets the pointing center and tracking mode based on a specified phase center. If the pointing is set to None, the telescope operates in drift-scan mode.

        Parameters
        ----------
        pointing_ra : float, optional
            Right ascension of the phase center in deg, default is None
        pointing_dec : float, optional
            Declination of the phase center in deg, default is None
        """

        # Check if pointings are either None (unpointed) or set to a real number
        def valid_pointing(x):
            return isinstance(x, numbers.Real) or x is None

        if not (valid_pointing(pointing_ra) and valid_pointing(pointing_dec)):
            raise TypeError(
                f"Right ascension of {pointing_ra} or declination of {pointing_dec} is not a real number."
            )

        # Drift-scan mode
        if pointing_dec is None and pointing_ra is None:
            self.tracking = False
            self.d = None
            self.set_element_property("tracking", False, True)
            return

        # Tracking mode
        if pointing_dec % 90 == 0:
            pointing_ra = 0  # Ra can be set arbitrarily for dec = +- 90
        if pointing_ra is None:
            raise ValueError(
                "Please set both a pointing declination and a pointing right ascension, right ascension is now None."
            )
        self.tracking = True
        self.d = {"dec": pointing_dec, "ra": pointing_ra}
        self.set_element_property("tracking", True, True)
        self.set_element_property("d", self.d, True)

    def update_station_pointing(self, new_pointing_ra=None, new_pointing_dec=None):
        """
        Updates station_pointing (including tile pointing). If the pointing is set to None, the telescope operates in drift-scan mode.

        Parameters
        ----------
        pointing_ra : float, optional
            Right ascension of the phase center in deg, default is None
        pointing_dec : float, optional
            Declination of the phase center in deg, default is None
        """
        self._set_pointing_center(new_pointing_ra, new_pointing_dec)
        self.set_element_property("d", self.d, same_value=True)

    def reset_elements(self):
        """
        Resets all elements in the tile to the common pointing and unit gain.
        """
        [element.reset_elements() for element in self.elements]
        self.set_element_property("g", 1.0 + 0j, True)

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
        local_east = np.cross(local_north, normal_vector)

        return np.array([local_east, local_north, normal_vector])

    def set_ENU_positions(self):
        """
        Sets ENU (East-North-Up) coordinates.
        """
        [
            tile.set_ENU_positions(self.ENU_rotation_matrix(), self.p)
            for tile in self.elements
        ]

    def set_array_position(self, station):
        """
        Sets position of station within the interferometric array compared to another station (so the position is the ENU position of self in the reference frame of the other station)

        Parameters
        ----------
        station : Station object
            reference station
        """
        self.p_array = np.dot(station.ENU_rotation_matrix(), self.p - station.p)

    def get_element_property(self, property):
        """
        Helper function for the element list. Retrieves a property of the underlying elements.
        """
        return np.array([getattr(element, property) for element in self.elements])

    def set_element_property(self, property, values, same_value=False):
        """
        Helper function for the element list. Sets a property of the underlying elements.
        """
        if same_value:
            [setattr(element, property, values) for element in self.elements]
        else:
            [
                setattr(element, property, value)
                for element, value in zip(self.elements, values)
            ]

    def calculate_array_factor(
        self, directions, pointing_directions, frequency, tile_beams=None
    ):
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
        pointing_directions : ndarray
            3xM array of unit length vectors that decribe the direction of the pointing center in ENU coordinates.

        Returns
        -------
        ndarray
            M length response
        """
        k = 2 * np.pi * frequency / const.c.value

        # Make sure the directions are fed correctly. They must be floats for the jit array factor to work
        directions = np.array(directions, dtype=float).reshape(3, -1)

        # Adjust for the phase center
        if self.tracking:
            pointing_directions = np.array(pointing_directions, dtype=float).reshape(
                3, -1
            )
            directions -= pointing_directions

        # Calculate the gemetric response of the tile elements (array factor)
        tile_factors = calculate_array_factor_contribution(
            positions=self.get_element_property("p_ENU"),
            gains=self.get_element_property("g"),
            k=k,
            directions=directions,
        )

        # Sum over the tiles
        if tile_beams is None:
            station_beam = np.mean(tile_factors, axis=0)
        else:
            tile_beams = np.array(tile_beams)
            tile_responses = tile_beams * tile_factors
            station_beam = np.mean(tile_responses, axis=0)

        return station_beam

    def calculate_response(
        self,
        directions,
        frequency,
        antenna_mode=None,
        pointing_directions=None,
        calculate_all_tiles=True,
    ):
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
        pointing_directions : ndarray, optional
            3xM array of unit length vectors that decribe the direction of the pointing center in ENU coordinates. Default is None (drift-scan)
        calculate_all_tiles : bool, optional
            If set, all tile beams are computed separately (allows for antenna drift), if unset, a tile beam pattern is reused for all tiles. Default is False.

        Returns
        -------
        ndarray
            M length response
        """
        if antenna_mode is not None:
            # Calculate antenna beams (only needs to be done once for now, since all antenna beam patterns are the same)
            antenna_beams = (
                self.elements[0]
                .elements[0]
                .calculate_response(
                    frequency=frequency,
                    directions=directions,
                    mode=antenna_mode,
                )
            )
        else:
            antenna_beams = None

        # Calculate the geometric delays of the antennas to get the full tile beams
        if calculate_all_tiles:
            # all tiles are calculated separately (allows for antenna drift)
            tile_beams = [
                tile.calculate_response(
                    directions=directions,
                    pointing_directions=pointing_directions,
                    frequency=frequency,
                    antenna_beams=antenna_beams,
                )
                for tile in self.elements
            ]
        else:
            # We use a set tile beam with unit gain per antenna for all tiles. The positions are based on the first tile
            model_tile = copy.deepcopy(self.elements[0])
            model_tile.reset_elements()
            tile_beam = model_tile.calculate_response(
                directions=directions,
                pointing_directions=pointing_directions,
                frequency=frequency,
                antenna_beams=antenna_beams,
            )
            tile_beams = [tile_beam for _ in self.elements]

        # Combine the tiles with geometric delay
        station_beam = self.calculate_array_factor(
            directions=directions,
            pointing_directions=pointing_directions,
            frequency=frequency,
            tile_beams=tile_beams,
        )

        return self.g * station_beam

    @staticmethod
    def _skycoord_ENU(sky_coord, obs_time, station_location):
        # Transform to AltAz frame
        altaz = sky_coord.transform_to(
            AltAz(obstime=obs_time, location=station_location)
        )
        altitude = altaz.alt.rad
        azimuth = altaz.az.rad

        # Exception for sources beneath the horizon
        if altitude < 0:
            return np.array([np.nan, np.nan, np.nan])

        # Convert AltAz to ENU
        east = np.cos(altitude) * np.sin(azimuth)
        north = np.cos(altitude) * np.cos(azimuth)
        up = np.sin(altitude)

        return np.array([east, north, up])

    def radec_to_ENU(
        self,
        time,
        right_ascension=None,
        declination=None,
        temporal_offset=0,
        number_of_timesteps=1,
        tracking_direction=False,
    ):
        """
        Calculates a sky direction in the stations ENU frame at a given time

        Parameters
        ----------
        time : str
            Start time of observation in UTC format (YYYY-MM-DDThh:mm:ss, example: 2024-07-04T19:25:00)
        right_ascension : float, optional
            RA in deg, default is None (for if the phase center is observed)
        declination : float, optional
            dec in deg, default is None (for if the phase center is observed)
        temporal_offset : float
            Time resolution in s.
        number_of_timesteps : int
            Number of simulated time intervals, default is 1
        tracking_direction : bool
            If set, compute the direction of the phase center instead of the Ra,Dec. Default is False.

        Returns
        -------
        ndarray
            source direction unit vector in ENU
        """

        if tracking_direction:
            if self.tracking:
                right_ascension = self.d["ra"]
                declination = self.d["dec"]
            else:
                return np.zeros([3, number_of_timesteps])

        station_location = EarthLocation.from_geocentric(
            x=self.p[0], y=self.p[1], z=self.p[2], unit="m"
        )

        base_time = Time(time, location=station_location)
        base_LST = base_time.sidereal_time("mean").deg
        hour_angles = np.array(
            [
                base_LST + n * temporal_offset / 240 - right_ascension
                for n in range(number_of_timesteps)
            ]
        )
        hour_angles *= np.pi / 180
        lattitude = station_location.lat.rad
        declination *= np.pi / 180

        azimuths = np.pi + np.arctan2(
            np.sin(hour_angles),
            np.cos(hour_angles) * np.sin(lattitude)
            - np.tan(declination) * np.cos(lattitude),
        )
        altitudes = np.arcsin(
            np.sin(lattitude) * np.sin(declination)
            + np.cos(lattitude) * np.cos(declination) * np.cos(hour_angles)
        )

        # Exception for sources beneath the horizon
        altitudes[altitudes < 0] = np.nan

        # Convert AltAz to ENU
        east = np.cos(altitudes) * np.sin(azimuths)
        north = np.cos(altitudes) * np.cos(azimuths)
        up = np.sin(altitudes)

        return np.array([east, north, up])
