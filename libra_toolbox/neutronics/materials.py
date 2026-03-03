import openmc


def get_exp_cllif_density(temp, LiCl_frac=0.695):
    """Calculates density of ClLiF [g/cc] from temperature in Celsius
    and molar concentration of LiCl. Valid for 660 C - 1000 C.
    Source:
    G. J. Janz, R. P. T. Tomkins, C. B. Allen;
    Molten Salts: Volume 4, Part 4
    Mixed Halide Melts Electrical Conductance, Density, Viscosity, and Surface Tension Data.
    J. Phys. Chem. Ref. Data 1 January 1979; 8 (1): 125–302.
    https://doi.org/10.1063/1.555590
    """
    temp = temp + 273.15  # Convert temperature from Celsius to Kelvin
    C = LiCl_frac * 100  # Convert molar concentration to molar percent

    a = 2.25621
    b = -8.20475e-3
    c = -4.09235e-4
    d = 6.37250e-5
    e = -2.52846e-7
    f = 8.73570e-9
    g = -5.11184e-10

    rho = a + b * C + c * temp + d * C**2 + e * C**3 + f * temp * C**2 + g * C * temp**2

    return rho


# Define Materials
# Source: PNNL Materials Compendium April 2021
# PNNL-15870, Rev. 2
Inconel625 = openmc.Material(name="Inconel 625")
Inconel625.set_density("g/cm3", 8.44)
Inconel625.add_element("C", 0.000990, "wo")
Inconel625.add_element("Al", 0.003960, "wo")
Inconel625.add_element("Si", 0.004950, "wo")
Inconel625.add_element("P", 0.000148, "wo")
Inconel625.add_element("S", 0.000148, "wo")
Inconel625.add_element("Ti", 0.003960, "wo")
Inconel625.add_element("Cr", 0.215000, "wo")
Inconel625.add_element("Mn", 0.004950, "wo")
Inconel625.add_element("Fe", 0.049495, "wo")
Inconel625.add_element("Co", 0.009899, "wo")
Inconel625.add_element("Ni", 0.580000, "wo")
Inconel625.add_element("Nb", 0.036500, "wo")
Inconel625.add_element("Mo", 0.090000, "wo")

# alumina insulation
# data from https://precision-ceramics.com/materials/alumina/
Alumina = openmc.Material(name="Alumina insulation")
Alumina.add_element("O", 0.6, "ao")
Alumina.add_element("Al", 0.4, "ao")
Alumina.set_density("g/cm3", 3.98)

# epoxy
Epoxy = openmc.Material(name="Epoxy")
Epoxy.add_element("C", 0.70, "wo")
Epoxy.add_element("H", 0.08, "wo")
Epoxy.add_element("O", 0.15, "wo")
Epoxy.add_element("N", 0.07, "wo")
Epoxy.set_density("g/cm3", 1.2)

# helium @5psig
pressure = 34473.8  # Pa ~ 5 psig
temperature = 300  # K
R_he = 2077  # J/(kg*K)
density = pressure / (R_he * temperature)  # in kg/cm^3
density *= 1 / 1000  # in g/cm^3
Helium = openmc.Material(name="Helium")
Helium.add_element("He", 1.0, "ao")
Helium.set_density("g/cm3", density)

# PbLi - eutectic - natural - pure
Pbli = openmc.Material(name="pbli")
Pbli.add_element("Pb", 84.2, "ao")
Pbli.add_element("Li", 15.2, "ao")
Pbli.set_density("g/cm3", 11)

# lif-licl - eutectic - natural - pure
Cllif = openmc.Material(name="ClLiF")
LiCl_frac = 0.695  # at.fr.

Cllif.add_element("F", 0.5 * (1 - LiCl_frac), "ao")
Cllif.add_element("Li", 0.5 * (1 - LiCl_frac) + 0.5 * LiCl_frac, "ao")
Cllif.add_element("Cl", 0.5 * LiCl_frac, "ao")
Cllif.set_density("g/cm3", get_exp_cllif_density(650))  # 69.5 at. % LiCL at 650 C

# lif-licl - eutectic - natural - EuF3 spiced
Spicyclif = openmc.Material(name="spicyclif")
Spicyclif.add_element("F", 0.15935, "wo")
Spicyclif.add_element("Li", 0.17857, "wo")
Spicyclif.add_element("Cl", 0.6340, "wo")
Spicyclif.add_element("Eu", 0.0279, "wo")

# FLiNaK - eutectic - natural - pure
Flinak = openmc.Material(name="flinak")
Flinak.add_element("F", 50, "ao")
Flinak.add_element("Li", 23.25, "ao")
Flinak.add_element("Na", 5.75, "ao")
Flinak.add_element("K", 21, "ao")
Flinak.set_density("g/cm3", 2.020)


# Aluminum : 2.6989 g/cm3
Aluminum = openmc.Material()
Aluminum.set_density("g/cm3", 2.6989)
Aluminum.add_nuclide("Al27", 1.0, "ao")

# Name: Air
# Density : 0.001205 g/cm3
# Reference: None
# Describes: All atmospheric, non-object chambers
Air = openmc.Material(name="Air")
Air.set_density("g/cm3", 0.001205)
Air.add_element("C", 0.00015, "ao")
Air.add_nuclide("N14", 0.784431, "ao")
Air.add_nuclide("O16", 0.210748, "ao")
Air.add_nuclide("Ar40", 0.004671, "ao")

# Name: Portland concrete
# Density: 2.3 g/cm3
# Reference: PNNL Report 15870 (Rev. 1)
# Describes: facility foundation, floors, walls
Concrete = openmc.Material()
Concrete.set_density("g/cm3", 2.3)
Concrete.add_nuclide("H1", 0.168759, "ao")
Concrete.add_element("C", 0.001416, "ao")
Concrete.add_nuclide("O16", 0.562524, "ao")
Concrete.add_nuclide("Na23", 0.011838, "ao")
Concrete.add_element("Mg", 0.0014, "ao")
Concrete.add_nuclide("Al27", 0.021354, "ao")
Concrete.add_element("Si", 0.204115, "ao")
Concrete.add_element("K", 0.005656, "ao")
Concrete.add_element("Ca", 0.018674, "ao")
Concrete.add_element("Fe", 0.004264, "ao")

# Name: Portland iron concrete
# Density: 3.8 g/cm3 as roughly measured using scale and assuming rectangular prism
# Reference: PNNL Report 15870 (Rev. 1)
# Describes: Potential new walls, shielding doors
IronConcrete = openmc.Material()
IronConcrete.set_density("g/cm3", 3.8)
IronConcrete.add_nuclide("H1", 0.135585, "ao")
IronConcrete.add_nuclide("O16", 0.150644, "ao")
IronConcrete.add_element("Mg", 0.002215, "ao")
IronConcrete.add_nuclide("Al27", 0.005065, "ao")
IronConcrete.add_element("Si", 0.013418, "ao")
IronConcrete.add_element("S", 0.000646, "ao")
IronConcrete.add_element("Ca", 0.040919, "ao")
IronConcrete.add_nuclide("Mn55", 0.002638, "ao")
IronConcrete.add_element("Fe", 0.648869, "ao")

#
# Lead : 11.34 g/cm3
Lead = openmc.Material()
Lead.set_density("g/cm3", 11.34)
Lead.add_nuclide("Pb204", 0.014, "ao")
Lead.add_nuclide("Pb206", 0.241, "ao")
Lead.add_nuclide("Pb207", 0.221, "ao")
Lead.add_nuclide("Pb208", 0.524, "ao")

# Name: Borated Polyethylene (5% B in via B4C additive)
# Density: 0.95 g/cm3
# Reference: PNNL Report 15870 (Rev. 1) but revised to make it 5 wt.% B
# Describes: General purpose neutron shielding
BPE = openmc.Material()
BPE.set_density("g/cm3", 0.95)
BPE.add_nuclide("H1", 0.1345, "wo")
BPE.add_element("B", 0.0500, "wo")
BPE.add_element("C", 0.8155, "wo")

# High Density Polyethylene
# Reference:  PNNL Report 15870 (Rev. 1)
HDPE = openmc.Material(name="HDPE")
HDPE.set_density("g/cm3", 0.95)
HDPE.add_element("H", 0.143724, "wo")
HDPE.add_element("C", 0.856276, "wo")

# Soil material taken from PNNL Materials Compendium for Earth, U.S. Average
Soil = openmc.Material(name="Soil")
Soil.set_density("g/cm3", 1.52)
Soil.add_element("O", 0.670604, percent_type="ao")
Soil.add_element("Na", 0.005578, percent_type="ao")
Soil.add_element("Mg", 0.011432, percent_type="ao")
Soil.add_element("Al", 0.053073, percent_type="ao")
Soil.add_element("Si", 0.201665, percent_type="ao")
Soil.add_element("K", 0.007653, percent_type="ao")
Soil.add_element("Ca", 0.026664, percent_type="ao")
Soil.add_element("Ti", 0.002009, percent_type="ao")
Soil.add_element("Mn", 0.000272, percent_type="ao")
Soil.add_element("Fe", 0.021050, percent_type="ao")

# Brick material taken from "Brick, Common Silica" from the PNNL Materials Compendium
# PNNL-15870, Rev. 2
Brick = openmc.Material(name="Brick")
Brick.set_density("g/cm3", 1.8)
Brick.add_element("O", 0.663427, percent_type="ao")
Brick.add_element("Al", 0.003747, percent_type="ao")
Brick.add_element("Si", 0.323229, percent_type="ao")
Brick.add_element("Ca", 0.007063, percent_type="ao")
Brick.add_element("Fe", 0.002534, percent_type="ao")

RicoRad = openmc.Material(name="RicoRad")
RicoRad.set_density("g/cm3", 0.945)
RicoRad.add_element("H", 0.14, percent_type="wo")
RicoRad.add_element("C", 0.84, percent_type="wo")
RicoRad.add_element("B", 0.02, percent_type="wo")

# LIBRA Materials
Steel = openmc.Material(name="Steel")
Steel.add_element("C", 0.005, "wo")
Steel.add_element("Fe", 0.995, "wo")
Steel.set_density("g/cm3", 7.82)

# Stainless Steel 304 from PNNL Materials Compendium (PNNL-15870 Rev2)
SS304 = openmc.Material(name="Stainless Steel 304")
# SS304.temperature = 700 + 273
SS304.add_element("C", 0.000800, "wo")
SS304.add_element("Mn", 0.020000, "wo")
SS304.add_element("P", 0.000450, "wo")
SS304.add_element("S", 0.000300, "wo")
SS304.add_element("Si", 0.010000, "wo")
SS304.add_element("Cr", 0.190000, "wo")
SS304.add_element("Ni", 0.095000, "wo")
SS304.add_element("Fe", 0.683450, "wo")
SS304.set_density("g/cm3", 8.00)

# Using Microtherm with 1 a% Al2O3, 27 a% ZrO2, and 72 a% SiO2
# https://www.foundryservice.com/product/microporous-silica-insulating-boards-mintherm-microtherm-1925of-grades/
Firebrick = openmc.Material(name="Firebrick")
# Estimate average temperature of Firebrick to be around 300 C
# Firebrick.temperature = 273 + 300
Firebrick.add_element("Al", 0.004, "ao")
Firebrick.add_element("O", 0.666, "ao")
Firebrick.add_element("Si", 0.240, "ao")
Firebrick.add_element("Zr", 0.090, "ao")
Firebrick.set_density("g/cm3", 0.30)

# Using 2:1 atom ratio of LiF to BeF2, similar to values in
# Seifried, Jeffrey E., et al. â€˜A General Approach for Determination of
# Acceptable FLiBe Impurity Concentrations in Fluoride-Salt Cooled High
# Temperature Reactors (FHRs)â€™. Nuclear Engineering and Design, vol. 343, 2019,
# pp. 85â€“95, https://doi.org10.1016/j.nucengdes.2018.09.038.
# Also using natural lithium enrichment (~7.5 a% Li6)
Flibe_nat = openmc.Material(name="Flibe_nat")
# Flibe_nat.temperature = 700 + 273
Flibe_nat.add_element("Be", 0.142857, "ao")
Flibe_nat.add_nuclide("Li6", 0.021685, "ao")
Flibe_nat.add_nuclide("Li7", 0.264029, "ao")
Flibe_nat.add_element("F", 0.571429, "ao")
Flibe_nat.set_density("g/cm3", 1.94)

Copper = openmc.Material(name="Copper")
# Estimate copper temperature to be around 100 C
# Copper.temperature = 100 + 273
Copper.add_element("Cu", 1.0, "ao")
Copper.set_density("g/cm3", 8.96)

Beryllium = openmc.Material(name="Beryllium")
# Estimate Be temperature to be around 100 C
# Be.temperature = 100 + 273
Beryllium.add_element("Be", 1.0, "ao")
Beryllium.set_density("g/cm3", 1.848)

# Heater
Heater_mat = openmc.Material(name="heater")
Heater_mat.add_element("C", 0.000990, "wo")
Heater_mat.add_element("Al", 0.003960, "wo")
Heater_mat.add_element("Si", 0.004950, "wo")
Heater_mat.add_element("P", 0.000148, "wo")
Heater_mat.add_element("S", 0.000148, "wo")
Heater_mat.add_element("Ti", 0.003960, "wo")
Heater_mat.add_element("Cr", 0.215000, "wo")
Heater_mat.add_element("Mn", 0.004950, "wo")
Heater_mat.add_element("Fe", 0.049495, "wo")
Heater_mat.add_element("Co", 0.009899, "wo")
Heater_mat.add_element("Ni", 0.580000, "wo")
Heater_mat.add_element("Nb", 0.036500, "wo")
Heater_mat.add_element("Mo", 0.090000, "wo")
Heater_mat.set_density("g/cm3", 2.44)

# Lithium-Lead
# Composition from certificate of analysis provided with Lithium-Lead from Camex
Lithium_lead = openmc.Material(name="Lithium Lead")
Lithium_lead.add_element("Pb", 0.993479, "wo")
Lithium_lead.add_element("Li", 0.0064, "wo")
Lithium_lead.add_element("Tl", 0.00002, "wo")
Lithium_lead.add_element("Zn", 0.000002, "wo")
Lithium_lead.add_element("Sn", 0.000002, "wo")
Lithium_lead.add_element("Sb", 0.000002, "wo")
Lithium_lead.add_element("Ni", 0.000001, "wo")
Lithium_lead.add_element("Cu", 0.000002, "wo")
Lithium_lead.add_element("Cd", 0.000002, "wo")
Lithium_lead.add_element("Bi", 0.00008, "wo")
Lithium_lead.add_element("As", 0.000002, "wo")
Lithium_lead.add_element("Ag", 0.000008, "wo")
Lithium_lead.set_density("g/cm3", 9.10411395)  # Density at 600C

# 316L Stainless Steel
# Data from https://www.thyssenkrupp-materials.co.uk/stainless-steel-316l-14404.html
SS316L = openmc.Material(name="316L Steel")
SS316L.add_element("C", 0.0003, "wo")
SS316L.add_element("Si", 0.01, "wo")
SS316L.add_element("Mn", 0.02, "wo")
SS316L.add_element("P", 0.00045, "wo")
SS316L.add_element("S", 0.000151, "wo")
SS316L.add_element("Cr", 0.175, "wo")
SS316L.add_element("Ni", 0.115, "wo")
SS316L.add_element("N", 0.001, "wo")
SS316L.add_element("Mo", 0.00225, "wo")
SS316L.add_element("Fe", 0.655599, "wo")
