# building the MIT-VaultLab neutron generator
# angular and energy distribution

import inspect
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


def _unit(uvw):
    """Returns ``uvw`` normalised, raising for a zero or non-finite vector."""
    uvw = np.asarray(uvw, dtype=float)
    norm = np.linalg.norm(uvw)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"reference_uvw must be a nonzero finite vector, got {uvw}")
    return uvw / norm


def _orthogonal_unit_vector(uvw):
    """Returns a unit vector orthogonal to ``uvw``."""
    uvw = _unit(uvw)
    # start from the axis that is the least aligned with uvw
    axis = np.zeros(3)
    axis[np.argmin(np.abs(uvw))] = 1.0
    vwu = axis - axis.dot(uvw) * uvw
    return vwu / np.linalg.norm(vwu)


def _polar_azimuthal(mu, phi, reference_uvw):
    """Builds an openmc.stats.PolarAzimuthal for any OpenMC version.

    OpenMC 0.15.3 added the ``reference_vwu`` argument (the direction the
    azimuthal angle is measured from) and it defaults to (1, 0, 0). A
    ``reference_uvw`` parallel to that default, for instance (1, 0, 0), is
    rejected with a ValueError. Only in that case do we pass an orthogonal
    ``reference_vwu``, so every other axis keeps OpenMC's default and gives
    exactly the same particles as before. The azimuthal angle is uniform, so
    the source is statistically unchanged either way.
    """
    uvw = _unit(reference_uvw)
    kwargs = {}
    if "reference_vwu" in inspect.signature(openmc.stats.PolarAzimuthal).parameters:
        # same parallel test as OpenMC, against its default reference_vwu
        if np.linalg.norm(np.cross([1.0, 0.0, 0.0], uvw)) <= 1e-6:
            kwargs["reference_vwu"] = _orthogonal_unit_vector(uvw)
    return openmc.stats.PolarAzimuthal(
        mu=mu, phi=phi, reference_uvw=reference_uvw, **kwargs
    )


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
        angle = _polar_azimuthal(mu, phi, reference_uvw)
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
