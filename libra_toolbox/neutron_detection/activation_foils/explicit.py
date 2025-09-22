from .settings import *
from .calculations import n93_number, delay_time
import numpy as np


def get_chain(irradiations, decay_constant=Nb92m_decay_constant):
    """
    Returns the value of
    (1 - exp(-\lambda * \Delta t_1)) * (1 - exp(-\lambda * \Delta t_2)) * ... * (1 - exp(-\lambda * \Delta t_n))
    where \Delta t_i is the time between the end of the i-th period (rest or irradiation) and the start of the next one

    Args:
        irradiations (list): list of dictionaries with keys "t_on" and "t_off" for irradiations

    Returns:
        float or pint.Quantity: the value of the chain
    """
    result = 1
    periods = [{"start": irradiations[0]["t_on"], "end": irradiations[0]["t_off"]}]
    for irr in irradiations[1:]:
        periods.append({"start": periods[-1]["end"], "end": irr["t_on"]})
        periods.append({"start": irr["t_on"], "end": irr["t_off"]})

    for period in periods:
        delta_t = period["end"] - period["start"]
        result = 1 - result * np.exp(-decay_constant * delta_t)
    return result


def get_neutron_flux(experiment: dict, irradiations: list):
    """calculates the neutron flux during the irradiation

    Args:
        experiment (dict): dictionary containing the experiment data
        irradiations (list): list of dictionaries with keys "t_on" and "t_off" for irradiations

    Returns:
        pint.Quantity: neutron flux
    """
    overall_efficiency = (
        (geometric_efficiency * nal_gamma_efficiency * branching_ratio)
        * ureg.count
        / ureg.particle
    )
    number_of_Nb92m_decays_measured = experiment["photon_counts"] / overall_efficiency

    number_of_Nb92m_decays_measured *= 0.5  # times by 0.5 because of double counting

    flux = (
        number_of_Nb92m_decays_measured
        / n93_number(experiment["foil_mass"])
        / Nb93_n_2n_Nb92m_cross_section_at_14Mev
    )

    flux *= get_chain(irradiations) ** -1
    time_between_generator_off_and_start_of_counting = delay_time(
        experiment["time_generator_off"], experiment["start_time_counting"]
    )
    flux *= (
        -1
        / Nb92m_decay_constant
        * (
            np.exp(
                -Nb92m_decay_constant
                * (
                    time_between_generator_off_and_start_of_counting
                    + experiment["counting_time"]
                )
            )
            - np.exp(
                -Nb92m_decay_constant * time_between_generator_off_and_start_of_counting
            )
        )
    ) ** -1

    # convert n/cm2/s to n/s
    area_of_sphere = 4 * np.pi * experiment["distance_from_center_of_target_plane"] ** 2

    flux *= area_of_sphere

    return flux


def get_neutron_flux_error(experiment: dict):
    """
    Returns the uncertainty of the neutron flux as a pint.Quantity

    Args:
        experiment (dict): dictionary containing the experiment data
        irradiations (list): list of dictionaries with keys "t_on" and "t_off" for irradiations

    Returns:
        pint.Quantity: uncertainty of the neutron flux
    """
    error_counts = experiment["photon_counts_uncertainty"] / experiment["photon_counts"]
    error_mass = 0.0001 * ureg.g / experiment["foil_mass"]
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
