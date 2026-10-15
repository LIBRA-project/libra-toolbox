import openmc
from components import Breeder, HeadSpace, Vessel1L
from assemblies import BreederVessel1LAssembly
from materials import Flibe_nat, Helium, Inconel625
from vault import build_vault_model

# Materials
vessel_material = Inconel625
breeder_material = Flibe_nat
headspace_material = Helium

# Components
vessel = Vessel1L(
    name="Vessel1L",
    material=vessel_material,
)

breeder = Breeder(
    name="Breeder",
    density=1.94,  # g/cm^3
    mass=1880,  # g
    material=breeder_material,
)

headspace = HeadSpace(
    name="HeadSpace",
    material=headspace_material,
)

# Assembly
assembly = BreederVessel1LAssembly(
    name="BreederVessel1LAssembly",
    vessel=vessel,
    breeder=breeder,
    headspace=headspace,
)

# # Build OpenMC cells
# cells = assembly.geometry()

# # OpenMC geometry
# geometry = openmc.Geometry(cells)

# # model
# model = openmc.model.Model(geometry=geometry)
# model.export_to_model_xml()

settings = openmc.Settings()
tallies = openmc.Tallies()


model = build_vault_model(
    settings=settings,
    tallies=tallies,
    experiment_placements=[
        {
            "assembly": assembly,
            "position": (500.0, 300.0, 10.16),
            "rotation": (0.0, 0.0, 45.0),
            "envelope_radius": 30.0,
        }
    ],
)

model.export_to_model_xml()