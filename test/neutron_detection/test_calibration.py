from libra_toolbox.neutron_detection.activation_foils.calibration import get_decay_lines


def test_get_decay_liness():
    decay_lines = get_decay_lines(['Na22', 'Cs137'])
    assert isinstance(decay_lines, dict)
    assert 'energy' in decay_lines['Na22'].keys()