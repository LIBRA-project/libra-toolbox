import openmc


def get_exp_cllif_density(temp, LiCl_frac=0.695):
    """ Calculates density of ClLiF [g/cc] from temperature in Celsius
    and molar concentration of LiCl. Valid for 660 C - 1000 C. 
    Source: 
    G. J. Janz, R. P. T. Tomkins, C. B. Allen; 
    Molten Salts: Volume 4, Part 4 
    Mixed Halide Melts Electrical Conductance, Density, Viscosity, and Surface Tension Data. 
    J. Phys. Chem. Ref. Data 1 January 1979; 8 (1): 125–302.
    https://doi.org/10.1063/1.555590
    """
    temp = temp + 273.15 # Convert temperature from Celsius to Kelvin
    C = LiCl_frac * 100  # Convert molar concentration to molar percent

    a = 2.25621
    b = -8.20475e-3
    c = -4.09235e-4
    d = 6.37250e-5
    e = -2.52846e-7
    f = 8.73570e-9
    g = -5.11184e-10

    rho = a + b * C + c * temp + d * C**2 \
        + e * C**3 + f * temp * C**2 + g * C * temp**2


    return rho


# Define Materials
# Source: PNNL Materials Compendium April 2021
# PNNL-15870, Rev. 2
inconel625 = openmc.Material(name='Inconel 625')
inconel625.set_density('g/cm3', 8.44)
inconel625.add_element('C',  0.000990, 'wo')
inconel625.add_element('Al', 0.003960, 'wo')
inconel625.add_element('Si', 0.004950, 'wo')
inconel625.add_element('P',  0.000148, 'wo')
inconel625.add_element('S',  0.000148, 'wo')
inconel625.add_element('Ti', 0.003960, 'wo')
inconel625.add_element('Cr', 0.215000, 'wo')
inconel625.add_element('Mn', 0.004950, 'wo')
inconel625.add_element('Fe', 0.049495, 'wo')
inconel625.add_element('Co', 0.009899, 'wo')
inconel625.add_element('Ni', 0.580000, 'wo')
inconel625.add_element('Nb', 0.036500, 'wo')
inconel625.add_element('Mo', 0.090000, 'wo')

# Stainless Steel 304 from PNNL Materials Compendium (PNNL-15870 Rev2)
SS304 = openmc.Material(name="Stainless Steel 304")
# SS304.temperature = 700 + 273
SS304.add_element('C',  0.000800, "wo")
SS304.add_element('Mn', 0.020000, "wo")
SS304.add_element('P',  0.000450 , "wo")
SS304.add_element('S',  0.000300, "wo")
SS304.add_element('Si', 0.010000, "wo")
SS304.add_element('Cr', 0.190000, "wo")
SS304.add_element('Ni', 0.095000, "wo")
SS304.add_element('Fe', 0.683450, "wo")
SS304.set_density("g/cm3", 8.00)

# Using Microtherm with 1 a% Al2O3, 27 a% ZrO2, and 72 a% SiO2
# https://www.foundryservice.com/product/microporous-silica-insulating-boards-mintherm-microtherm-1925of-grades/
firebrick = openmc.Material(name="Firebrick")
# Estimate average temperature of Firebrick to be around 300 C
# Firebrick.temperature = 273 + 300
firebrick.add_element('Al', 0.004, 'ao')
firebrick.add_element('O', 0.666, 'ao')
firebrick.add_element('Si', 0.240, 'ao')
firebrick.add_element('Zr', 0.090, 'ao')
firebrick.set_density('g/cm3', 0.30)

# alumina insulation
# data from https://precision-ceramics.com/materials/alumina/
alumina = openmc.Material(name='Alumina insulation')
alumina.add_element('O', 0.6, 'ao')
alumina.add_element('Al', 0.4, 'ao')
alumina.set_density('g/cm3', 3.98)

# air
air = openmc.Material(name="Air")
air.add_element("C", 0.00012399 , 'wo')
air.add_element('N', 0.75527, 'wo')
air.add_element('O', 0.23178, 'wo')
air.add_element('Ar', 0.012827, 'wo')
air.set_density('g/cm3', 0.0012)

# epoxy
epoxy = openmc.Material(name='Epoxy')
epoxy.add_element('C', 0.70, 'wo')
epoxy.add_element('H', 0.08, 'wo')
epoxy.add_element('O', 0.15, 'wo')
epoxy.add_element('N', 0.07, 'wo') 
epoxy.set_density('g/cm3', 1.2)  

# helium @5psig
pressure = 34473.8  # Pa ~ 5 psig 
temperature = 300  # K
R_he = 2077  # J/(kg*K)
density = pressure / (R_he * temperature) # in kg/cm^3
density *= 1 / 1000 # in g/cm^3
he = openmc.Material(name="Helium")
he.add_element('He', 1.0, 'ao')
he.set_density('g/cm3', density)

# PbLi - natural - pure
pbli = openmc.Material(name="pbli")
pbli.add_element("Pb", 84.2, "ao")
pbli.add_element("Li", 15.2, "ao")
pbli.set_density("g/cm3", 11)

flibe = openmc.Material(name="flibe")
flibe.add_element("Li", 2.0, "ao")
flibe.add_element("Be", 1.0, "ao")
flibe.add_element("F", 4.0, "ao")
flibe.set_density("g/cm3", 1.94)

# lif-licl - natural - pure
cllif_nat = openmc.Material(name='ClLiF natural')
LiCl_frac = 0.695  # at.fr.

cllif_nat.add_element('F', .5*(1 - LiCl_frac), 'ao')
cllif_nat.add_element('Li', 1.0, 'ao')
cllif_nat.add_element('Cl', .5*LiCl_frac, 'ao')
cllif_nat.set_density('g/cm3', get_exp_cllif_density(650)) # 69.5 at. % LiCL at 650 C

# lif-licl - natural - EuF3 spiced
spicyclif = openmc.Material(name="spicyclif")
spicyclif.add_element("F", .15935, "wo")
spicyclif.add_element("Li", .17857, "wo")
spicyclif.add_element("Cl", .6340, "wo")
spicyclif.add_element("Eu", .0279, "wo")

# FLiNaK - natural - pure
flinak = openmc.Material(name="flinak")
flinak.add_element("F", 50, "ao")
flinak.add_element("Li", 23.25, "ao")
flinak.add_element("Na", 5.75, "ao")
flinak.add_element("K", 21, "ao")
flinak.set_density("g/cm3", 2.020)
