import numpy as np


class Source:
    def __init__(
        self,
        right_ascension,
        declination,
        brightness,
        reference_frequency,
        spectral_index,
        logSI,
    ):
        """
        Class to describe a sky-model source. The brightness is computed as for a LOFAR sourcedb (see https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb)

        Parameters
        ----------
        right_ascension : str
            RA in HH:MM:SS.SSS
        declination : str
            Dec in +-DD.MM.SS.SSS
        brightness : float
            Source brightness (I0)
        reference_frequency : float
            v0
        spectral_index : str
            str in the form [c0,c1,c2]
        logSI : bool
            whether the spectral index is logarithmic
        """
        self.ra = self.parse_right_ascension(right_ascension)
        self.dec = self.parse_declination(declination)
        if logSI:
            self.I = lambda v: self.logarithmic_spectral_index_brightness(
                v, brightness, reference_frequency, spectral_index
            )
        else:
            self.I = lambda v: self.linear_spectral_index_brightness(
                v, brightness, reference_frequency, spectral_index
            )

    @staticmethod
    def parse_right_ascension(right_ascension_string):
        """
        Helper function to convert the RA to degrees
        """
        right_ascension_string = right_ascension_string.strip(" ")
        HA, min, sec = right_ascension_string.split(":")

        # Check for negative angles
        sign = -1 if HA[0] == "-" else 1

        # 1 hour 15 degrees (15*24=360)
        right_ascension_degrees = abs(int(HA)) * 15 + int(min) / 4 + float(sec) / 240
        return sign * right_ascension_degrees

    @staticmethod
    def parse_declination(declination_string):
        """
        Helper function to convert the Dec to degrees
        """
        declination_string = declination_string.replace(" ", "")
        declination_string = declination_string.strip("+")
        fields = declination_string.split(".")
        deg = int(fields[0])
        min = int(fields[1])
        sec = float(".".join(fields[2:]))  # Accepts either an integer or a float
        declination_degrees = abs(deg) + min / 60 + sec / 3600
        return np.sign(deg) * declination_degrees

    @staticmethod
    def logarithmic_spectral_index_brightness(
        frequency, brightness, reference_frequency, spectral_index
    ):
        """
        Helper function to calculate the brightness as a function of frequency for a logarithmic spectral index
        """
        spectral_index = spectral_index.strip("[]").split(",")
        x = frequency / reference_frequency
        spectral_shape = x ** (
            sum(float(c) * np.log10(x) ** i for i, c in enumerate(spectral_index))
        )
        return brightness * spectral_shape

    @staticmethod
    def linear_spectral_index_brightness(
        frequency, brightness, reference_frequency, spectral_index
    ):
        """
        Helper function to calculate the brightness as a function of frequency for a linear spectral index
        """
        spectral_index = spectral_index.strip("[]").split(",")
        x = frequency / reference_frequency - 1
        spectral_shape = sum(float(c) * x**i for i, c in enumerate(spectral_index))
        return brightness + spectral_shape


class Patch:
    def __init__(
        self,
        patch_right_ascension,
        patch_declination,
    ):
        """
        Class that contains a calibration patch (group/cluster of sources)

        Parameters
        ----------
        patch_right_ascension : str
            RA in HH:MM:SS.SSS
        patch_declination : str
            Dec in +-DD.MM.SS.SSS
        """
        self.elements = {}
        self.ra = Source.parse_right_ascension(patch_right_ascension)
        self.dec = Source.parse_declination(patch_declination)

    def add_source(
        self,
        source_name,
        right_ascension,
        declination,
        brightness,
        reference_frequency,
        spectral_index,
        logSI,
    ):
        """
        Creates a source within the Patch. The brightness is computed as for a LOFAR sourcedb (see https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb)

        Parameters
        ----------
        right_ascension : str
            RA in HH:MM:SS.SSS
        declination : str
            Dec in +-DD.MM.SS.SSS
        brightness : float
            Source brightness (I0)
        reference_frequency : float
            v0
        spectral_index : str
            str in the form [c0,c1,c2]
        logSI : bool
            whether the spectral index is logarithmic
        """
        self.elements[source_name] = Source(
            right_ascension=right_ascension,
            declination=declination,
            brightness=brightness,
            reference_frequency=reference_frequency,
            spectral_index=spectral_index,
            logSI=logSI,
        )


class Skymodel:
    def __init__(self, filename):
        """
        Reads a skymodel from disk and converts it into a SHIMMERR sky model.

        Parameters
        ----------
        filename : str
            path + filename of the sky model
        """
        self.elements = {}
        with open(filename, "r") as f:
            self.parse_formatstring(f.readline())
            for inputline in f:
                inputline = inputline.rstrip("\n")

                # Skip comments
                if inputline.startswith("#") or inputline.replace(",", "") == "":
                    continue
                inputfields = inputline.split(",")

                # For readability
                def get_item(itemname):
                    try:
                        return inputfields[self.items[itemname]].strip(" ")
                    except IndexError:
                        return ""

                # New patch
                patch_name = get_item("patch")
                if patch_name not in self.elements.keys():
                    # Check if this is actually a patch definition or a sneaky source
                    if get_item("i") != "":
                        raise ValueError(
                            f"Patch {patch_name} has not been defined before adding sources (or you have specified a brightness for this patch)."
                        )
                    # Add a new patch
                    self.elements[patch_name] = Patch(
                        patch_right_ascension=get_item("ra"),
                        patch_declination=get_item("dec"),
                    )

                else:
                    # Check if a reference frequency has been given or if we use the default
                    try:
                        reference_frequency = float(get_item("referencefrequency"))
                    except ValueError:
                        reference_frequency = self.items["default_referencefrequency"]

                    # Fill in whether the spectral index is logarithmic
                    try:
                        logSI = (
                            get_item("logarithmicsi") == "true"
                        )  # can't cast to bool, that just checks whether the string is empty
                    except KeyError:
                        logSI = True

                    # Add the source
                    self.elements[patch_name].add_source(
                        source_name=get_item("name"),
                        right_ascension=get_item("ra"),
                        declination=get_item("dec"),
                        brightness=float(get_item("i")),
                        reference_frequency=reference_frequency,
                        spectral_index=get_item("spectralindex"),
                        logSI=logSI,
                    )

    def parse_formatstring(self, string):
        # Anything not in this list will be ignored
        list_of_itemnames = [
            "name",
            "patch",
            "ra",
            "dec",
            "i",
            "referencefrequency",
            "spectralindex",
            "logarithmicsi",
        ]

        # remove capitalization, spaces and 'format=' and split the string into fields
        string = string.lower()
        string = string.replace(" ", "")
        fields = string.split(",")

        # Create a dictionary with the indexes of the fields
        self.items = {}
        for itemname in list_of_itemnames:
            # If a value is not in the file, we skip it
            try:
                matching_string = list(filter(lambda x: itemname in x, fields))[0]
            except IndexError:
                if itemname == "spectralindex" or itemname == "logarithmicsi":
                    continue
                raise KeyError(f"Item {itemname} missing from sky model")
            self.items[itemname] = fields.index(matching_string)

            # There is a default value for the reference frequency that we also need to save
            if itemname == "referencefrequency":
                try:
                    _, default_reference_frequency = matching_string.split("=")
                    default_reference_frequency = "".join(
                        i
                        for i in default_reference_frequency
                        if i.isdigit()
                        or i == "e"
                        or i == "-"  # e and - are needed for numbers like 5e-1
                    )
                    self.items["default_referencefrequency"] = float(
                        default_reference_frequency
                    )
                except ValueError:
                    print("No default reference frequency provided in skymodel.")
                    continue

    def join_patches(self, patch_names, joint_name=None, joint_ra=None, joint_dec=None):
        # Make sure metadata of the new patch is provided
        if joint_name is None:  # If no name, just join the patch names
            joint_name = "_".join(patch_names)
        if (
            joint_ra is None
        ):  # If no ra or dec, just take the metadata of the first patch
            joint_ra = self.elements[patch_names[0]].ra
        if joint_dec is None:
            joint_dec = self.elements[patch_names[0]].dec

        # Create the new joint patch (still devoid of sources)
        self.elements[joint_name] = Patch(
            patch_right_ascension=joint_ra,
            patch_declination=joint_dec,
        )

        for patch in patch_names:
            # Add the sources from each patch
            self.elements[joint_name].elements.update(self.elements[patch].elements)

            # Then delete the old patch
            del self.elements[patch]
