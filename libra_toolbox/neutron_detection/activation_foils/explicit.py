from .settings import *
from .foils import Experiment, Foil
from .calculations import n93_number, delay_time
import numpy as np
import pandas as pd
import os
import warnings


def get_chain(irradiations, decay_constant):

    """ Returns the value of
    (1 - exp(-\lambda * \Delta t_1)) * (1 - exp(-\lambda * \Delta t_2)) * ... * (1 - exp(-\lambda * \Delta t_n))
    where \Delta t_i is the time between the end of the i-th period (rest or irradiation) and the start of the next one

    Args:
        irradiations (list): list of dictionaries with keys "t_on" and "t_off" for irradiations

    Returns:
        float or pint.Quantity: the value of the chain """
    
    result = 1
    periods = [{"start": irradiations[0]["t_on"], "end": irradiations[0]["t_off"]}]
    for irr in irradiations[1:]:
        periods.append({"start": periods[-1]["end"], "end": irr["t_on"]})
        periods.append({"start": irr["t_on"], "end": irr["t_off"]})

    for period in periods:
        delta_t = period["end"] - period["start"]
        result = 1 - result * np.exp(-decay_constant * delta_t)
    return result


def get_efficiency(energies=None, coeff=None,
                   coeff_energy_bounds=None,
                   coeff_type='total', geometric_eff=1.0):
    """Calculates the total efficiency of a gamma detector
    based on provided coefficients. These coefficients could
    be for calculating the total efficiency (coeff_type='total')
    or for calculating the intrinsic efficiency (coeff_type='intrinsic')"""

    if coeff is None:
        # These are the efficiencies reported for activation foil analysis
        # of the BABY 100 mL runs
        geometric_eff = 0.5
        # Account for double counting in these experiments
        intrinsic_efficiency = 0.344917296922981 * 2
        total_efficiency = geometric_eff * intrinsic_efficiency
        warnings.warn('Using NaI efficiency from BABY 100 mL runs, which may not reflect detector efficiency accurately.')
    else:
        # Check that efficiency is being interpolated between the bounds of the fit curve
        ergs = energies.to(ureg.keV).magnitude
        erg_bounds = coeff_energy_bounds.to(ureg.keV).magnitude
        if np.min(ergs) < np.min(erg_bounds) or np.max(ergs) > np.max(erg_bounds):
            warnings.warn('Efficiency is being extrapolated according to efficiency fit curve bounds.')
        if coeff_type.lower()=='total':
            total_efficiency = np.polyval(coeff, ergs)
        elif coeff_type.lower()=='intrinsic':
            intrinsic_efficiency = np.polyval(coeff, ergs)
            total_efficiency = geometric_eff * intrinsic_efficiency
    total_efficiency *= ureg.count / ureg.particle
    return total_efficiency


def get_neutron_flux(experiment: Experiment, irradiations: list, foil: Foil):
    """calculates the neutron flux during the irradiation
    Based on Equation 1 from:
    Lee, Dongwon, et al. "Determination of the Deuterium-Tritium (D-T) Generator 
    Neutron Flux using Multi-foil Neutron Activation Analysis Method." , 
    May. 2019. https://doi.org/10.2172/1524045

    Args:
        experiment (dict): dictionary containing the experiment data
        irradiations (list): list of dictionaries with keys "t_on" and "t_off" for irradiations

    Returns:
        pint.Quantity: neutron flux
    """


    # time_between_generator_off_and_start_of_counting = delay_time(
    #     experiment["time_generator_off"], experiment["start_time_counting"]
    # )
    time_between_generator_off_and_start_of_counting = (
        experiment.start_time_counting - experiment.time_generator_off
    ).total_seconds() * ureg.second

    if experiment.total_eff_coeff is not None:
        total_efficiency = get_efficiency(energies=foil.photon_energies,
                                          coeff=experiment.total_eff_coeff,
                                          coeff_energy_bounds=experiment.efficiency_bounds,
                                          coeff_type='total')
    elif experiment.intrinsic_eff_coeff is not None:
        total_efficiency = get_efficiency(energies=foil.photon_energies,
                                          coeff=experiment.intrinsic_eff_coeff,
                                          coeff_energy_bounds=experiment.efficiency_bounds,
                                          coeff_type='intrinsic',
                                          geometric_eff=experiment.geometric_efficiency)
    else:
        total_efficiency = get_efficiency()

    #Spectroscopic Factor to account for the branching ratio and the
    # total detection efficiency

    f_spec = total_efficiency * foil.branching_ratio

    number_of_decays_measured = experiment.photon_counts / f_spec

    # print('photon counts ', experiment.photon_counts)
    # print('total efficiency ', total_efficiency)
    # print('branching ratio ', foil.branching_ratio)

    # print('photon counts / total efficiency / branching ratio: ', number_of_decays_measured)
    # print('number ', foil.atoms)
    # print('cross section ', foil.cross_section)


    flux = (
        number_of_decays_measured
        / foil.atoms
        / foil.cross_section
    )

    # print('# of decays measured / n_Nb93 / xs: ', flux)


    f_time = (get_chain(irradiations, foil.decay_constant)
                * np.exp( -foil.decay_constant
                     * time_between_generator_off_and_start_of_counting) 
                * (1 - np.exp( -foil.decay_constant * experiment.real_count_time))
                * (experiment.live_count_time / experiment.real_count_time)
                / foil.decay_constant
    )

    # print('decay constant ', foil.decay_constant)
    # print('get_chain() ', get_chain(irradiations, foil.decay_constant))
    # print('time between generator off and start of counting ', time_between_generator_off_and_start_of_counting)
    # print('flux / f_time :', (flux / f_time).to(1/ureg.s/ureg.cm**2))

    # Correction factor of gamma-ray self-attenuation in the foil
    if foil.thickness is None:
        f_self = 1
    else:
        f_self = ( (1 - 
                        np.exp(-foil.mass_attenuation_coefficient
                            * foil.density
                            * foil.thickness))
                    / (foil.mass_attenuation_coefficient
                    * foil.density
                    * foil.thickness)
        ).to('dimensionless')


    flux /= (f_time * f_self)

    # print('flux / f_time / f_self: ', flux.to(1/ureg.s/ureg.cm**2))


    # print('flux: ', flux.to_reduced_units())
    # print('f_time', f_time)
    # print('f_self', f_self)

    # convert n/cm2/s to n/s
    area_of_sphere = 4 * np.pi * experiment.distance_from_center_of_target_plane ** 2


    flux *= area_of_sphere

    flux = flux.to(1/ureg.s)

    # print('final flux: ', flux, '\n')

    return flux


def get_neutron_flux_error(experiment: Experiment, foil: Foil):
    """
    Returns the uncertainty of the neutron flux as a pint.Quantity

    Args:
        experiment (dict): dictionary containing the experiment data
        irradiations (list): list of dictionaries with keys "t_on" and "t_off" for irradiations

    Returns:
        pint.Quantity: uncertainty of the neutron flux
    """ 
    error_counts = experiment.photon_counts_uncertainty / experiment.photon_counts
    error_mass = 0.0001 * ureg.g / foil.mass
    error_geometric_eff = 0.025 / geometric_efficiency
    error_intrinsic_eff = 0.025 / nal_gamma_efficiency

    error = np.sqrt(
        error_counts**2
        + error_mass**2
        + error_geometric_eff**2
        + error_intrinsic_eff**2
    )
    return error.to(ureg.dimensionless).magnitude


if __name__ == "__main__":
    pass
