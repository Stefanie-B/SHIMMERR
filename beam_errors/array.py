import numpy as np
import numbers
from joblib import Parallel, delayed
from numba import jit, complex128, float64, prange
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta
from astropy import constants as const
from astropy import units as u


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

    relative_pointing = np.empty(directions.shape, dtype=np.float64)
    for i in prange(directions.shape[0]):
        relative_pointing[i, :] = directions[i, :] - pointing[i]
    phase_delay = k * np.dot(positions, relative_pointing)

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
        k = 2 * np.pi * frequency / const.c.value

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
        Resets all elements in the tile to the common pointing and unit gain
        """
        [
            element.update_tile(new_pointing=self.d, new_gain=1)
            for element in self.elements
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
        k = 2 * np.pi * frequency / const.c.value

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

    def radec_to_ENU(self, right_ascension, declination, time, temporal_offset=None):
        """
        Calculates a sky direction in the stations ENU frame at a given time

        Parameters
        ----------
        right_ascension : float
            RA in deg
        declination : float
            dec in deg
        time : str
            Observing time in UTC format (YYYY-MM-DDThh:mm:ss, example: 2024-07-04T19:25:00)
        temporal_offset : None or float
            Time offset from given time in s.

        Returns
        -------
        ndarray
            source direction unit vector in ENU
        """
        # Get the source position, time and station position as astropy objects
        sky_coord = SkyCoord(
            ra=right_ascension * u.deg, dec=declination * u.deg, frame="icrs"
        )
        obs_time = Time(time)
        if temporal_offset is not None:
            obs_time += TimeDelta(temporal_offset, format="sec")
        station_location = EarthLocation(
            x=self.p[0] * u.m, y=self.p[1] * u.m, z=self.p[2] * u.m
        )

        # Transform to AltAz frame
        altaz = sky_coord.transform_to(
            AltAz(obstime=obs_time, location=station_location)
        )
        altitude = altaz.alt.rad
        azimuth = altaz.az.rad

        # Exception forsources beneath the horizon
        if altitude < 0:
            return np.array([np.nan, np.nan, np.nan])

        # Convert AltAz to ENU
        east = np.cos(altitude) * np.sin(azimuth)
        north = np.cos(altitude) * np.cos(azimuth)
        up = np.sin(altitude)

        return np.array([east, north, up])
