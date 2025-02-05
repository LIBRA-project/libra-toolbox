from settings import *
# from .calculations import n93_number, delay_time
import numpy as np
import pandas as pd
import os


def get_foil_data(foil: dict, 
                  filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nuclide_data.xlsx'),
                  xslib='EAF2010'):
    
    # Read in nuclide data
    df = pd.read_excel(filepath, skiprows=1)
    
    # Only get info for one nuclide
    mask = df['Nuclide'] == foil['nuclide']

    # Calculate number of nuclide atoms in foil
    num_element = (foil['mass'] 
                   / (df['Element_Atomic_Mass'][mask].item() * ureg.g / ureg.mol)
                   * (6.022e23 * ureg.particle / ureg.mol)
    )
    
    foil['number'] = num_element * df['Natural_Abundance'][mask].item()

    # Get density and mass attenuation coefficient
    foil['density'] = df['Density'][mask].item() * ureg.g / ureg.cm**3
    foil['mass_attenuation_coefficient'] = df['Mass_Attenuation_Coefficient'][mask].item() * (ureg.cm**2/(ureg.g))

    # Get cross section for available reactions
    for reaction in foil['reactions'].keys():
        heading = reaction + '_' + xslib + '_xs'
        if heading in df.keys():
            foil['reactions'][reaction]['cross_section'] = df[heading][mask].item() * ureg.barn

    return foil



def get_chain(irradiations, decay_constant):
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


def get_neutron_flux(experiment: dict, irradiations: list, foil: dict, 
                     reaction: str):
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

    print('inside get_neutron_flux()')
    decay_constant = foil['reactions'][reaction]['decay_constant']
    foil = get_foil_data(foil)

    # time_between_generator_off_and_start_of_counting = delay_time(
    #     experiment["time_generator_off"], experiment["start_time_counting"]
    # )
    time_between_generator_off_and_start_of_counting = (
        experiment["start_time_counting"] - experiment["time_generator_off"]
    ).seconds * ureg.second

    overall_efficiency = (
        (experiment[reaction]['efficiency'] * foil['reactions'][reaction]['branching_ratio'])
        * ureg.count
        / ureg.particle
    )

    print('efficiency', overall_efficiency)
    number_of_decays_measured = experiment[reaction]["photon_counts"] / overall_efficiency
    print('number of decays measured', number_of_decays_measured)
    print('number', foil['number'])
    print('cross section', foil['reactions'][reaction]['cross_section'])


    flux = (
        number_of_decays_measured
        / foil['number']
        / foil['reactions'][reaction]['cross_section']
    )

    f_time = (get_chain(irradiations, foil['reactions'][reaction]['decay_constant'])
                * np.exp( -decay_constant
                     * time_between_generator_off_and_start_of_counting) 
                * (1 - np.exp( -decay_constant * experiment['real_count_time']))
                * (experiment['live_count_time'] / experiment['real_count_time'])
                / decay_constant
    )

    print('att coeff', foil['mass_attenuation_coefficient'])
    print('density', foil['density'])
    print('thickness', foil['thickness'])

    f_self = ( (1 - 
                    np.exp(-foil['mass_attenuation_coefficient']
                           * foil['density']
                           * foil['thickness']))
                / (foil['mass_attenuation_coefficient'] 
                   * foil['density']
                   * foil['thickness'])
    ).to('dimensionless')

    print('flux: ', flux.to_reduced_units())
    print('f_time', f_time)
    print('f_self', f_self)

    flux /= (f_time * f_self)
    


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
