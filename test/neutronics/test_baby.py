import openmc
from libra_toolbox.neutronics import vault
from libra_toolbox.neutronics.baby1l import baby_geometry
from libra_toolbox.neutronics.neutron_source import A325_generator_diamond
from libra_toolbox.neutronics.materials import *


def baby_model():
    """Returns an openmc model of the BABY experiment.

    Returns:
        the openmc model
    """

    materials = [
        Inconel625,
        Cllif,
        SS304,
        Heater_mat,
        Firebrick,
        Alumina,
        Lead,
        Air,
        Epoxy,
        Helium,
        HDPE,
    ]

    # BABY coordinates
    x_c = 587  # cm
    y_c = 60  # cm
    z_c = 100  # cm
    sphere, cllif_cell, cells = baby_geometry(x_c, y_c, z_c)

    ############################################################################
    # Define Settings

    settings = openmc.Settings()

    src = A325_generator_diamond((x_c, y_c, z_c - 5.635), (1, 0, 0))
    settings.source = src
    settings.batches = 3
    settings.inactive = 0
    settings.run_mode = "fixed source"
    settings.particles = int(100)
    settings.output = {"tallies": False}
    settings.photon_transport = False

    ############################################################################
    overall_exclusion_region = -sphere

    ############################################################################
    # Specify Tallies
    tallies = openmc.Tallies()

    tbr_tally = openmc.Tally(name="TBR")
    tbr_tally.scores = ["(n,Xt)"]
    tbr_tally.filters = [openmc.CellFilter(cllif_cell)]
    tallies.append(tbr_tally)

    model = vault.build_vault_model(
        settings=settings,
        tallies=tallies,
        added_cells=cells,
        added_materials=materials,
        overall_exclusion_region=overall_exclusion_region,
    )

    return model


if __name__ == "__main__":
    model = baby_model()
    model.run()
    sp = openmc.StatePoint(f"statepoint.{model.settings.batches}.h5")
    tbr_tally = sp.get_tally(name="TBR").get_pandas_dataframe()

    print(f"TBR: {tbr_tally['mean'].iloc[0]:.6e}\n")
    print(f"TBR std. dev.: {tbr_tally['std. dev.'].iloc[0]:.6e}\n")