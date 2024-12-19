from collections.abc import Iterable
from libra_toolbox.neutronics.neutron_source import A325_generator_diamond

import pytest


def test_get_avg_neutron_rate():
    try:
        import openmc
    except ImportError:
        pytest.skip("OpenMC is not installed")

    source = A325_generator_diamond((0, 0, 0), (0, 0, 1))

    assert isinstance(source, Iterable)
    for s in source:
        assert isinstance(s, openmc.IndependentSource)
