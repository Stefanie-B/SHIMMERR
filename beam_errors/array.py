import numpy as np

# For LOFAR specific
from lofarantpos.db import LofarAntennaDatabase
from lofarantpos import geo


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

    def __init__(self, position, spatial_delay=0, gain=1.0 + 0j):
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
        self.d = spatial_delay

    def calc_delay(self, pointing, reference_position=[0, 0, 0]):
        relative_position = self.p - reference_position
        self.d = relative_position @ pointing

    def update_antenna(self, new_delay=None, new_gain=None):
        """
        Updates antenna weight based on a new pointing vector or new gain
        """
        if new_delay is not None:
            self.d = new_delay
        if new_gain is not None:
            self.g = new_gain

    def update_common_settings(self, new_d=None, old_d=0, new_g=None, old_g=0 + 0j):
        if new_d is not None:
            new_delay = self.d - old_d + new_d
        else:
            new_delay = None
        if new_g is not None:
            new_gain = self.g - old_g + new_g
        else:
            new_g = None
        self.update_antenna(new_delay, new_gain)


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

    def __init__(self, positions, central_position, pointing, gain=1.0 + 0j):
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
        relative_position = self.p - reference_position
        self.d = relative_position @ pointing
        self.d = spatial_delay
        self.g = gain
        self.elements = [Antenna(position, self.d, self.g) for position in positions]

    def calc_delay(self, pointing, reference_position=[0, 0, 0]):
        relative_position = self.p - reference_position
        self.d = relative_position @ pointing

    def update_tile(self, new_delay, new_gain):
        """
        Updates common tile gain or pointing.

        The new gains of the elements are set as:
        new_element_gain = old_element_gain - old_common_gain + new_common_gain
        This retains perturbations set on the elements
        """
        [
            element.update_common_settings(
                new_d=new_delay, old_d=self.d, new_g=new_gain, old_g=self.g
            )
            for element in self.elements
        ]
        self.g = new_gain
        self.d = new_delay

    def reset_elements(self):
        """
        Resets all elements in the tile to the common gain and pointing (removing individual perturbations)
        """
        [
            element.update_antenna(new_delay=self.d, new_gain=self.g)
            for element in self.elements
        ]


class Station:
    def __init__(self, positions, pointing=1.0 + 0j, gain=1.0 + 0j, tile_pointing=None):
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
        self.s = pointing
        if tile_pointing is None:
            self.tile_pointing = self.s
        self.g = gain

        self.p = np.mean(positions, axis=(0, 1))

        self.elements = [
            Tile(per_tile_positions, self.p, self.s, self.g)
            for per_tile_positions in positions
        ]

    def update_station(self, new_gain=None):
        """
        Updates common tile gain. The new gains of the elements are set as:
        new_element_gain = old_element_gain - old_common_gain + new_common_gain
        This retains perturbations set on the elements
        """
        [
            element.update_gain(new_gain=element.g - self.g + new_gain)
            for element in self.elements
        ]
        self.g = new_gain

    def reset_elements(self):
        """
        Resets all elements in the tile to the common gain and pointing (removing individual perturbations)
        """
        [
            element.update_tile(new_pointing=self.tile_pointing, new_gain=self.g)
            for element in self.elements
        ]


### TODO:
### - Find good way to do delay calc
### - Fix pointing flexible for tile pointing errors
### - Fix documentation (especially on update functions)
### - Write tests
