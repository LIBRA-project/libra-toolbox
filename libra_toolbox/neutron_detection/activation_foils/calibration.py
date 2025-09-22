from dataclasses import dataclass
from typing import List
import datetime
import numpy as np


@dataclass
class Nuclide:
    """
    Class to hold the information of a nuclide.

    Attributes
    ----------
    name :
        The name of the nuclide.
    energy :
        The energy of the gamma rays emitted by the nuclide (in keV).
    intensity :
        The intensity of the gamma rays emitted by the nuclide.
    half_life :
        The half-life of the nuclide in seconds.
    atomic_mass :
        The atomic mass of the nuclide in atomic mass units (amu).
    abundance :
        The natural abundance of the nuclide as a fraction (default is 1.00).
    """

    name: str
    energy: List[float] = None
    intensity: List[float] = None
    half_life: float = None
    atomic_mass: float = None
    abundance: float = 1.00

    @property
    def decay_constant(self):
        """
        Returns the decay constant of the nuclide in 1/s.
        """
        return np.log(2) / self.half_life


ba133 = Nuclide(
    name="Ba133",
    energy=[80.9979, 276.3989, 302.8508, 356.0129, 383.8485],
    intensity=[0.329, 0.0716, 0.1834, 0.6205, 0.0894],
    half_life=10.551 * 365.25 * 24 * 3600,
)
co60 = Nuclide(
    name="Co60",
    energy=[1173.228, 1332.492],
    intensity=[0.9985, 0.999826],
    half_life=1925.28 * 24 * 3600,
)
na22 = Nuclide(
    name="Na22",
    energy=[511, 1274.537],
    intensity=[1.80, 0.9994],
    half_life=2.6018 * 365.25 * 24 * 3600,
)
cs137 = Nuclide(
    name="Cs137",
    energy=[661.657],
    intensity=[0.851],
    half_life=30.08 * 365.25 * 24 * 3600,
)
mn54 = Nuclide(
    name="Mn54",
    energy=[834.848],
    intensity=[0.99976],
    half_life=312.20 * 24 * 3600,
)

nb92m = Nuclide(
    name="Nb92m",
    energy=[934.44],
    intensity=[0.9915],
    half_life=10.25 * 24 * 3600,
)

nb93 = Nuclide(
    name="Nb93",
    atomic_mass=92.90637,
    abundance=1.00
)

zr89 = Nuclide(
    name="Zr89",
    energy=[909.15],
    intensity = [0.9904],
    half_life=78.41 * 3600
)

zr90 = Nuclide(
    name="Zr90",
    atomic_mass=89.90469876,
    abundance=0.515
)


@dataclass
class Reaction:
    reactant: Nuclide
    product: Nuclide
    cross_section: float
    """
    Class to hold the information of a reaction.
    Attributes
    ----------
    reactant :
        The reactant of the reaction.
    product :
        The product of the reaction.
    cross_section :
        The cross section of the reaction in cm2.
    """

nb93_n2n = Reaction(
    reactant=nb93,
    product=nb92m,
    cross_section=0.45966e-24 # cm2 at 14.1 MeV from IRDF-II 2020
)

zr90_n2n = Reaction(
    reactant=zr90,
    product=zr89,
    cross_section=0.62389e-24 # cm2 at 14.1 MeV from IRDF-II 2020
)


@dataclass
class CheckSource:
    nuclide: Nuclide
    activity_date: datetime.date
    activity: float

    """
    Class to hold the information of a check source.
    Attributes
    ----------
    nuclide :
        The nuclide of the check source.
    activity_date :
        The date of the calibrated activity of the check source.
    activity :
        The activity of the check source in Bq.
    """

    def get_expected_activity(self, date: datetime.date) -> float:
        """
        Returns the expected activity of the check source at a given date.

        Args:
            date: the date to calculate the expected activity for.

        Returns:
            the expected activity of the check source in Bq
        """

        decay_constant = np.log(2) / self.nuclide.half_life

        # Convert date to datetime if needed
        if isinstance(self.activity_date, datetime.date) and not isinstance(
            self.activity_date, datetime.datetime
        ):

            activity_datetime = datetime.datetime.combine(
                self.activity_date, datetime.datetime.min.time()
            )
            # add a timezone
            activity_datetime = activity_datetime.replace(tzinfo=date.tzinfo)
        else:
            activity_datetime = self.activity_date

        time = (date - activity_datetime).total_seconds()
        act_expec = self.activity * np.exp(-decay_constant * time)
        return act_expec


@dataclass
class ActivationFoil:
    reaction: Reaction
    mass: float
    name: str
    density: float = None
    thickness: float = None

    """Class to hold the information of an activation foil.
    Attributes
    ----------
    reaction :
        The reaction that produces the nuclide.
    mass :
        The mass of the foil in grams.
    name :
        The name of the foil.
    density :
        The density of the foil in g/cm3.
    thickness :
        The thickness of the foil in cm.        
    """

    def __post_init__(self):
        if (self.thickness is None) != (self.density is None):
            raise ValueError("Thickness and density must either both be floats or both be None.")

    @property
    def nb_atoms(self) -> float:
        """
        Returns the number of atoms in the foil.
        """
        avogadro = 6.022e23  # part/mol
        return self.reaction.reactant.abundance * (self.mass / self.reaction.reactant.atomic_mass * avogadro)
