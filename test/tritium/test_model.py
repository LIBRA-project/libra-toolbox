from libra_toolbox.tritium.model import Model, ureg
import numpy as np
import pytest


@pytest.mark.parametrize("TBR", [0, 1e-2, 0.5, 1, 2])
def test_simple_case(TBR):
    # build
    model = Model(
        radius=2 * ureg.m,
        height=4 * ureg.m,
        TBR=TBR * ureg.particle * ureg.neutron**-1,
        k_top=2 * ureg.m * ureg.s**-1,
        k_wall=3 * ureg.m * ureg.s**-1,
        irradiations=[(0 * ureg.s, 10 * ureg.s), (60 * ureg.s, 70 * ureg.s)],
        neutron_rate=30 * ureg.neutron * ureg.s**-1,
    )

    # run
    model.run(100 * ureg.s)

    # test
    neutron_fluence = (
        sum([irr[1] - irr[0] for irr in model.irradiations]) * model.neutron_rate
    )
    expected_total_production = model.TBR * neutron_fluence
    computed_total_production = (
        model.integrated_release_top()[-1] + model.integrated_release_wall()[-1]
    )
    expected_total_production = expected_total_production.to(ureg.particle)
    computed_total_production = computed_total_production.to(ureg.particle)

    assert np.isclose(model.concentrations[-1], 0)
    assert np.isclose(expected_total_production, computed_total_production, rtol=1e-2)


def test_mass_transport_coeffs_time_dependent():

    def k_top(t):
        return 2 / (1 + t.magnitude) * ureg.m * ureg.s**-1

    def k_wall(t):
        return 3 / (1 + t.magnitude) * ureg.m * ureg.s**-1

    model = Model(
        radius=2 * ureg.m,
        height=4 * ureg.m,
        TBR=1 * ureg.particle * ureg.neutron**-1,
        k_top=k_top,
        k_wall=k_wall,
        irradiations=[(0 * ureg.s, 10 * ureg.s), (60 * ureg.s, 70 * ureg.s)],
        neutron_rate=30 * ureg.neutron * ureg.s**-1,
    )

    # run
    model.run(100 * ureg.s)
