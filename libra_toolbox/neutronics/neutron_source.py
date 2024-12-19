# building the MIT-VaultLab neutron generator
# angular and energy distribution

from pathlib import Path
from collections.abc import Iterable
import pandas as pd
import numpy as np

try:
    import h5py
    import openmc
    from openmc import IndependentSource
except ModuleNotFoundError:
    pass


def A325_generator_diamond(
    center=(0, 0, 0), reference_uvw=(0, 0, 1)
) -> "Iterable[IndependentSource]":
    """
    Builds the MIT-VaultLab A-325 neutron generator in OpenMC
    with data tabulated from John Ball and Shon Mackie characterization
    via diamond detectors

    Parameters
    ----------
    center : tuple, optional
        coordinate position of the source (it is a point source),
        by default (0, 0, 0)
    reference_uvw : tuple, optional
        direction for the polar angle (tuple or list of versors)
    it is the same for the openmc.PolarAzimuthal class
    more specifically, polar angle = 0 is the direction of the D accelerator
    towards the Zr-T target, by default (0, 0, 1)

    Returns
    -------
        list of openmc neutron sources with angular and energy distribution
        and total strength of 1
    """
    try:
        import h5py
        import openmc
    except ModuleNotFoundError:
        raise ModuleNotFoundError("openmc and h5py are required")

    filename = "A325_generator_diamond.h5"
    filename = str(Path(__file__).parent) / Path(filename)

    with h5py.File(filename, "r") as source:
        df = pd.DataFrame(source["values/table"][()]).drop(columns="index")
        # energy values
        energies = np.array(df["Energy (MeV)"]) * 1e6
        # angle column names
        angles = df.columns[1:]
        # angular bins in [0, pi)
        pbins = np.cos([np.deg2rad(float(a)) for a in angles] + [np.pi])
        spectra = [np.array(df[col]) for col in angles]

    # yield values for strengths
    yields = np.sum(spectra, axis=-1) * np.diff(pbins)
    yields /= np.sum(yields)

    # azimuthal values
    phi = openmc.stats.Uniform(a=0, b=2 * np.pi)

    all_sources = []
    for i, angle in enumerate(pbins[:-1]):

        mu = openmc.stats.Uniform(a=pbins[i + 1], b=pbins[i])

        space = openmc.stats.Point(center)
        angle = openmc.stats.PolarAzimuthal(mu=mu, phi=phi, reference_uvw=reference_uvw)
        energy = openmc.stats.Tabular(
            energies, spectra[i], interpolation="linear-linear"
        )
        strength = yields[i]

        my_source = openmc.IndependentSource(
            space=space,
            angle=angle,
            energy=energy,
            strength=strength,
            particle="neutron",
        )

        all_sources.append(my_source)

    return all_sources
