from libra_toolbox.neutronics.neutron_source import *
from collections.abc import Iterable

def test_get_avg_neutron_rate(tmpdir):

    source = A325_generator_diamond((0,0,0), (0,0,1))

    assert source == Iterable[openmc.Source]