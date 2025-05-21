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
    """

    name: str
    energy: List[float] = None
    intensity: List[float] = None
    half_life: float = None
    atomic_mass: float = None
    peak_finding_prms: dict = None

    def __post_init__(self):
        if self.peak_finding_prms is None:
            self.peak_finding_prms = {}

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
    peak_finding_prms={
        "start_index": 100,
        "prominence_rel": 0.1,
        "height_rel": 0.10,
        "width": [10, 200],
        "distance": 30,
    },
)
co60 = Nuclide(
    name="Co60",
    energy=[1173.228, 1332.492],
    intensity=[0.9985, 0.999826],
    half_life=1925.28 * 24 * 3600,
    peak_finding_prms={
        "start_index": 400,
        "height_rel": 0.60,
        "width": [10, 150],
        "distance": 30,
    },
)
na22 = Nuclide(
    name="Na22",
    energy=[511, 1274.537],
    intensity=[1.80, 0.9994],
    half_life=2.6018 * 365.25 * 24 * 3600,
    peak_finding_prms={
        "start_index": 100,
        "prominence_rel": 0.1,
        "height_rel": 0.1,
        "width": [10, 150],
        "distance": 30,
    },
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
    peak_finding_prms={
        "start_index": 100,
        "prominence_rel": 0.1,
        "height_rel": 0.60,
        "width": [10, 150],
        "distance": 30,
    },
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
)


@dataclass
class Reaction:
    reactant: Nuclide
    product: Nuclide
    cross_section: float


@dataclass
class CheckSource:
    nuclide: Nuclide
    activity_date: datetime.date
    activity: float

    def get_expected_activity(self, date: datetime.date) -> float:

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
    abundance: float = 1.0
    thickness: float = None

    @property
    def nb_atoms(self):
        avogadro = 6.022e23  # part/mol
        return self.abundance * (
            self.mass / self.reaction.reactant.atomic_mass * avogadro
        )
