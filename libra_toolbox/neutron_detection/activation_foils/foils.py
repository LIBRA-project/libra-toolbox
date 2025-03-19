import numpy as np
import datetime
from zoneinfo import ZoneInfo
from .settings import ureg


def convert_to_datetime(time:str, tzinfo=ZoneInfo('America/New_York')):
    if isinstance(time, str):
        datetime_obj = datetime.datetime.strptime(time, "%m/%d/%Y %H:%M:%S")
    elif isinstance(time, datetime.datetime):
        datetime_obj = time
    else:
        raise ValueError('Time is neither a string nor a datetime.datetime object.')
    # Check if timezone info is included
    if datetime_obj.tzinfo is None:
        # If not, use New York time as default
        datetime_obj = datetime_obj.replace(tzinfo=tzinfo)
    return datetime_obj


class Experiment:
    def __init__(self, time_generator_off,
                 start_time_counting,
                 distance_from_center_of_target_plane: ureg.Quantity,
                 real_count_time: ureg.Quantity,
                 live_count_time=None,
                 name=None, generator=None, run=None,
                 total_eff_coeff=None, intrinsic_eff_coeff=None,
                 efficiency_bounds=None,
                 geometric_efficiency=1,
                 photon_counts=None,
                 photon_counts_uncertainty=None,
                 tzinfo=ZoneInfo('America/New_York')):
        # Ensure times are all datetime objects with timezones
        self.time_generator_off = convert_to_datetime(time_generator_off)
        self.start_time_counting = convert_to_datetime(start_time_counting)
        self.distance_from_center_of_target_plane = distance_from_center_of_target_plane
        self.real_count_time = real_count_time
        if live_count_time:
            self.live_count_time = live_count_time
        else:
            self.live_count_time = real_count_time
        self.name = name
        self.generator = generator
        self.run = run

        self.total_eff_coeff = total_eff_coeff
        self.intrinsic_eff_coeff = intrinsic_eff_coeff
        self.geometric_efficiency = geometric_efficiency
        self.efficiency_bounds = efficiency_bounds
        self.photon_counts = photon_counts
        self.photon_counts_uncertainty = photon_counts_uncertainty

    



class Foil:
    def __init__(self, mass: float, thickness: float, name=''):
        """
        Base class for neutron activation foils.
        :param mass: Mass of the foil in grams
        :param thickness: Thickness of the foil in cm
        """
        self.mass = mass
        self.thickness = thickness
        self.name = name
        self.half_life = None
        self.decay_constant = None
        self.reaction = None
        self.branching_ratio = None
        self.photon_energies = None
        self.cross_section = None  # in barns (1 barn = 1e-24 cm^2)
        self.density = None  # in g/cm^3
        self.atomic_mass = None
        self.abundance = None
        self.mass_attenuation_coefficient = None  # in cm^2/g
    
    def get_properties(self):
        """Returns the properties of the foil."""
        return {
            "Name": self.name,
            "Mass (g)": self.mass,
            "Thickness (cm)": self.thickness,
            "Decay Constant (1/s)": self.decay_constant,
            "Reaction": self.reaction,
            "Branching Ratio": self.branching_ratio,
            "Photon Energies": self.photon_energies,
            "Cross Section (barns)": self.cross_section,
            "Density (g/cm³)": self.density,
            "Atomic Mass": self.atomic_mass,
            "Abundance": self.abundance,
            "Mass Attenuation Coefficient (cm²/g)": self.mass_attenuation_coefficient
        }
    
    def get_decay_constant(self):
        self.decay_constant = np.log(2) / self.half_life.to(ureg.s)

    def get_atoms(self):
        self.atoms = (self.abundance 
                      * (self.mass 
                         * (6.022141e23 * ureg.particle / ureg.mol) 
                         / self.atomic_mass))

class Niobium(Foil):
    def __init__(self, mass: float, thickness: float, name=''):
        super().__init__(mass, thickness, name=name)

        self.half_life = np.array([10.15]) * ureg.day
        self.get_decay_constant()

        self.reaction = ["Nb93(n,2n)Nb92m"]
        self.branching_ratio = np.array([0.9915])
        self.photon_energies = np.array([934.44]) * ureg.keV

        # Cross section from EAF-2010 at 14 MeV
        self.cross_section = np.array([0.4729]) * ureg.barn

        self.density = 8.582 * ureg.g / ureg.cm**3 
        self.atomic_mass = 92.90637 * ureg.g/ureg.mol
        # Natural abundance of reactant in analyzed foil reaction
        self.abundance = np.array([1.00])
        self.get_atoms()

        # Should update mass attenuation coefficient to be photon energy-dependent
        self.mass_attenuation_coefficient = np.array([0.06120]) * ureg.cm**2 / ureg.g   # at 1 MeV

    def get_100mL_params(self):
        # Adjust parameters to replicate previous 100 mL results

        # half life used in BABY 100 mL analysis
        self.half_life = np.array([10.25]) * ureg.day
        self.get_decay_constant()

        # branching ratio used in BABY 100 mL analysis
        self.branching_ratio = np.array([0.999])

        # cross section used in BABY 100 mL analysis
        self.cross_section = np.array([0.46]) * ureg.barn



class Zirconium(Foil):
    def __init__(self, mass: float, thickness: float, name=''):
        super().__init__(mass, thickness, name=name)
        self.half_life = np.array([78.41,
                                   78.41]) * ureg.hour
        self.get_decay_constant()

        self.reaction = ["Zr90(n,2n)Zr89", 
                         "Zr90(n,2n)Zr89"]
        self.branching_ratio = np.array([0.45, 
                                         0.9904])
        self.photon_energies = np.array([511, 
                                         909.15]) * ureg.keV

        # Cross sections from EAF-2010 at 14 MeV
        self.cross_section = np.array([0.5068,
                                       0.5068]) * ureg.barn
        # Cross sections from ENDF/B-VIII.0 at 14.1 MeV
        # self.cross_section = np.array([0.5382,
        #                                0.5382]) * ureg.barn
                              
        self.density = 6.505 * ureg.g / ureg.cm**3 
        self.atomic_mass = 91.222 * ureg.g / ureg.mol
        # Natural abundance of reactant in analyzed foil reaction
        self.abundance = np.array([0.5145,
                                   0.5145])
        self.get_atoms()

        #From NIST Xray Mass Attenuation Coefficients Table 3
        self.mass_attenuation_coefficient = np.array([0.06156,
                                                      0.08590]) * ureg.cm**2 / ureg.g  # at 1 MeV


def convert_dict_to_foils(data:dict, use_100mL_params=False):
    if data["element"] == "Nb":
        foil = Niobium(mass=data["foil_mass"],
                       thickness=0.01 * ureg.inch,
                       name=data["foil_name"])
        if use_100mL_params:
            foil.thickness = None
            foil.get_100mL_params()
    elif data["element"] == "Zr":
        foil = Zirconium(mass=data["foil_mass"],
                         thickness=0.005 * ureg.inch,
                         name=data["foil_name"])
        
    # Clean up data dictionary to not include foil class arguments
    experiment_dict = {}
    for key in data.keys():
        if key not in ["element", "foil_mass", "foil_name"]:
            experiment_dict[key] = data[key]
    experiment = Experiment(**experiment_dict)

    return experiment, foil


