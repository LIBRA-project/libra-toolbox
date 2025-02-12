def build_vault_model(
    settings=None,
    tallies=None,
    added_cells=[],
    added_materials=[],
    overall_exclusion_region=None,
) -> "openmc.model.Model":
    """
    Builds a complete OpenMC model for a simulation setup representing a
    shielding system for MIT Vault Laboratory.

    Parameters:
    ----------
    settings : openmc.Settings, optional
        An instance of `openmc.Settings` to configure simulation parameters.
        If not specified, default settings are used.

    tallies : openmc.Tallies, optional
        An instance of `openmc.Tallies` to define the tallies to be collected
        during the simulation. Default is an empty object.

    added_cells : list of openmc.Cell, optional
        A list of additional cells to include in the geometry.
        Useful for extending the model with custom objects.

    added_materials : list of openmc.Material, optional
        A list of additional materials to include in the model.
        This allows for the inclusion of non-default materials.

    overall_exclusion_region : openmc.Region, optional
        An optional region that defines areas to exclude in the construction
        of the vault geometry. Can be used to simulate voids or areas without
        specific materials.

    Returns:
    -------
    openmc.model.Model
        A complete OpenMC model instance, ready for simulation.
        Includes specified materials, geometry, settings, and tallies.

    Notes:
    -----
    - The model includes a wide range of standard materials, such as concrete,
      steel, lead, and borated polyethylene, along with a detailed geometry
      based on predefined surfaces.
    - The function automatically downloads ENDFB-8.0 cross-section data
      for neutrons using the `openmc_data_downloader` library.
    - If an `overall_exclusion_region` is provided, it will be incorporated
      to exclude specific parts of the geometry.
    """

    # optional dependency
    try:
        import openmc
        import openmc.model
        import os

        if "OPENMC_CROSS_SECTIONS" not in os.environ:
            import openmc_data_downloader as odd
    except ModuleNotFoundError:
        raise ModuleNotFoundError("openmc and openmc_data_downloader are required.")

    materials = openmc.Materials(
        [
            Aluminum,
            Material_2,
            Air,
            Concrete,
            IronConcrete,
            Material_6,
            Material_7,
            Material_8,
            Material_10,
            Lead,
            BPE,
            Polyethylene,
            HDPE,
            Material_22,
            Material_23,
            Material_30,
            Material_40,
            Soil,
            Brick,
            RicoRad,
            Steel,
            SS304,
            Firebrick,
            Flibe_nat,
            Copper,
            Be,
        ]
    )

    # Add materials from imported model
    materials += added_materials

    if "OPENMC_CROSS_SECTIONS" not in os.environ:
        materials.download_cross_section_data(
            libraries=["ENDFB-8.0-NNDC"],
            set_OPENMC_CROSS_SECTIONS=True,
            particles=["neutron"],
        )
    #
    # Definition of the spherical void/blackhole boundary
    Surface_95 = openmc.Sphere(x0=0.0, y0=0.0, z0=0.0, r=2500.0, boundary_type="vacuum")

    # 24
    Surface_24 = openmc.model.RectangularParallelepiped(
        -81.28, 1143.0, -99.38, 650.24, 0.0, 424.18
    )

    # with an angle of 2.8 degrees. The positive vector points towards the
    # lower-right (Southeast) corner of the geometry
    Surface_49 = openmc.Plane(a=0.99881, b=-0.04885, c=0.0, d=1046.099544)
    #
    # Outer surface definition of the foundation underneath all basement labs
    Surface_94 = openmc.model.RectangularParallelepiped(
        -1104.9, 1143.0, -99.38, 1898.99, -81.28, 0.0
    )

    #
    # The cuboid defining the outermost boundary of the Vault door in Room III
    Surface_13 = openmc.model.RectangularParallelepiped(
        0.51, 61.47, 268.62, 512.46, 0.0, 223.52
    )

    # The plane used to create the 30 degree north-most cut on the Vault door.
    # The positive vector points towards the lower-left
    Surface_14 = openmc.Plane(a=0.5, b=0.86603, c=0.0, d=446.3839386000001)

    # The plane used to create the 30 degree south-most cut on the Vault door
    # the positive vector points towards the upper-left
    Surface_15 = openmc.Plane(a=0.5, b=-0.86603, c=0.0, d=-227.45393860000007)

    # The main Vault shield door in Room III
    Vault_door_reg = -Surface_13 & -Surface_14 & -Surface_15

    #
    # North B-HDPE shield in entrance to Vault in Room III
    Surface_17 = openmc.model.RectangularParallelepiped(
        -38.1, 0.0, 466.4, 499.42, 10.16, 213.36
    )

    # The northern Ricorad extra Vault door shielding in Room III
    Vault_door_shield_n_pillar_reg = -Surface_17

    #
    # South B-HDPE shield in entrance to Vault in Room III
    Surface_18 = openmc.model.RectangularParallelepiped(
        -38.1, 0.0, 281.33, 314.35, 10.16, 213.36
    )

    # The southern Ricorad extra Vault door shielding in Room III
    Vault_door_shield_s_pillar_reg = -Surface_18

    #
    # Surface definition for west iron-brick pile around DANTE selection magnet
    Surface_10 = openmc.model.RectangularParallelepiped(
        636.75, 703.59, 412.74, 538.47, 10.16, 152.4
    )

    # The western DANTE beamline (Fe or Pb fill?) concrete block shield
    DANTE_vault_w_shield_reg = -Surface_10

    #
    # Surface definition for east iron-brick pile around DANTE selection magnet
    Surface_9 = openmc.model.RectangularParallelepiped(
        830.58, 878.84, 412.74, 538.47, 10.16, 135.89
    )

    # The eastern DANTE beamline (Fe or Pb fill?) concrete block shield
    DANTE_vault_e_shield_reg = -Surface_9

    #
    # 11
    Surface_11 = openmc.model.RightCircularCylinder(
        (753.11, 538.48, 111.76), 111.76, 15.24, axis="y"
    )

    #
    # 2
    Surface_22 = openmc.model.RectangularParallelepiped(
        594.3, 1014.3, 538.47, 568.95, 10.16, 363.22
    )

    # with an angle of 2.8 degrees. The positive vector points towards the
    # lower-right (Southeast) corner of the geometry
    Surface_48 = openmc.Plane(a=0.99881, b=-0.04885, c=0.0, d=964.9095439999999)

    # The CMU wall partially covering the north shield wall in Room III
    Vault_north_wall_ext_reg = -Surface_22 & -Surface_48 & +Surface_11

    # The foundation underneath all basement lab rooms
    Region_21 = -Surface_94 & -Surface_49

    #
    # 36
    Surface_36 = openmc.model.RectangularParallelepiped(
        0.0, 1150.0, -99.38, 0.0, 0.0, 363.22
    )

    # The south Vault shield wall in Room III
    South_vault_wall_reg = -Surface_36 & -Surface_49

    #
    # 16
    Surface_16 = openmc.model.RectangularParallelepiped(
        945.1, 1095.1, 0.0, 568.96, 0.0, 363.22
    )

    # The east Vault shield wall in Room III with Room II entrance cutout
    East_vault_wall_reg = -Surface_16 & +Surface_48 & -Surface_49

    #
    # 38
    Surface_38 = openmc.model.RectangularParallelepiped(
        -104.9, 45.1, 281.33, 499.42, 10.16, 213.36
    )

    #
    # 39
    Surface_39 = openmc.model.RectangularParallelepiped(
        -81.28, 0.0, -99.38, 650.24, 0.0, 363.22
    )

    # The west Vault shield wall in Room III with Vault entrance cutout
    West_vault_wall_reg = -Surface_39 & +Surface_38

    #
    # 37
    Surface_37 = openmc.model.RectangularParallelepiped(
        -81.28, 1148.72, -99.38, 650.24, 363.22, 424.18
    )

    # The top (roof) Vault shield wall in Room III
    Vault_ceiling_reg = -Surface_37 & -Surface_49

    #
    # 12
    Surface_12 = openmc.model.RectangularParallelepiped(
        64.77, 1064.77, 0.0, 568.96, 0.0, 10.16
    )

    # The bottom Vault floor in Room III
    Vault_floor_reg = -Surface_12 & -Surface_48

    # 23
    Surface_23 = openmc.model.RectangularParallelepiped(
        0.0, 1150.0, 568.96, 650.24, 0.0, 363.22
    )

    #
    # The cyclotron beamline cutout in the north Vault shield wall
    Surface_102 = openmc.model.RightCircularCylinder(
        (317.1, 568.96, 50.0), 81.28, 5.0, axis="y"
    )

    # The north Vault shield wall in Room III with beamline cutouts
    North_vault_wall_reg = -Surface_23 & -Surface_49 & +Surface_11 & +Surface_102

    #
    # 82
    Surface_82 = openmc.model.RectangularParallelepiped(
        31.0, 42.43, 39.37, 529.59, 276.86, 279.4
    )

    #
    # 83
    Surface_83 = openmc.model.RectangularParallelepiped(
        31.0, 42.43, 39.37, 529.59, 297.18, 299.72
    )

    #
    # 84
    Surface_84 = openmc.model.RectangularParallelepiped(
        35.45, 37.99, 39.37, 529.59, 276.86, 299.72
    )

    # The I-beam support the main Vault shield door in Room III
    I_beam_reg = -Surface_82 | -Surface_83 | -Surface_84

    #
    # Inner surface defining the top/bottom DANTE selection magnets
    Surface_28 = openmc.model.RightCircularCylinder(
        (753.11, 497.84, 99.7), 75.0, 20.95, axis="z"
    )

    #
    # Outer surface defining the bottom DANTE selection magnet
    Surface_30 = openmc.model.RightCircularCylinder(
        (753.11, 497.84, 99.7), 8.0, 32.0, axis="z"
    )

    # The bottom DANTE beamline selection magnet in Room III
    DANTE_vault_bot_magnet_reg = -Surface_30 & +Surface_28

    #
    # Outer surface defining the top DANTE selection magnet
    Surface_35 = openmc.model.RightCircularCylinder(
        (753.11, 497.84, 115.83), 8.0, 32.0, axis="z"
    )

    # The top DANTE beamline selection magnet in Room III
    DANTE_vault_top_magnet_reg = -Surface_35 & +Surface_28

    #
    # Surface definition for selection magnet cutout of surface #27
    Surface_21 = openmc.model.RectangularParallelepiped(
        720.97, 785.24, 472.52, 522.52, 99.7, 123.83
    )

    #
    # Surface definition of square box that contains DANTE selection magnets
    Surface_27 = openmc.model.RectangularParallelepiped(
        711.2, 795.02, 477.52, 518.16, 89.54, 133.99
    )

    #
    # Selection magnet SE support leg
    Surface_29 = openmc.model.RectangularParallelepiped(
        787.4, 795.02, 477.52, 485.14, 10.16, 88.9
    )

    #
    # Selection magnet NE support leg
    Surface_31 = openmc.model.RectangularParallelepiped(
        787.4, 795.02, 510.54, 518.16, 10.16, 88.9
    )

    #
    # Selection magnet SW support leg
    Surface_33 = openmc.model.RectangularParallelepiped(
        711.2, 718.82, 477.52, 485.14, 10.16, 88.9
    )

    #
    # Thin selection magnet table top plate
    Surface_34 = openmc.model.RectangularParallelepiped(
        711.2, 795.02, 477.52, 518.16, 88.9, 89.54
    )

    # The DANTE beamline selection magnet stand in Room III
    DANTE_vault_mag_stand_reg = (
        (-Surface_27 & +Surface_21)
        | -Surface_29
        | -Surface_31
        | -Surface_33
        | -Surface_34
    )

    Region_28 = (
        -Surface_24
        & -Surface_49
        & ~North_vault_wall_reg
        & ~Vault_north_wall_ext_reg
        & ~South_vault_wall_reg
        & ~West_vault_wall_reg
        & ~East_vault_wall_reg
        & ~Vault_ceiling_reg
        & ~Vault_floor_reg
        & ~Vault_door_reg
        & ~Vault_door_shield_n_pillar_reg
        & ~Vault_door_shield_s_pillar_reg
        & ~I_beam_reg
        & ~DANTE_vault_w_shield_reg
        & ~DANTE_vault_e_shield_reg
        & ~DANTE_vault_bot_magnet_reg
        & ~DANTE_vault_top_magnet_reg
        & ~DANTE_vault_mag_stand_reg
    )
    if overall_exclusion_region:
        Region_28 = Region_28 & ~overall_exclusion_region

    Vault_air_cell = openmc.Cell(fill=Air, region=Region_28)
    DANTE_vault_mag_stand_cell = openmc.Cell(
        fill=Aluminum, region=DANTE_vault_mag_stand_reg
    )
    DANTE_vault_top_magnet_cell = openmc.Cell(
        fill=Material_2, region=DANTE_vault_top_magnet_reg
    )
    DANTE_vault_bot_magnet_cell = openmc.Cell(
        fill=Material_2, region=DANTE_vault_bot_magnet_reg
    )
    I_beam_cell = openmc.Cell(fill=Material_6, region=I_beam_reg)
    North_vault_wall_cell = openmc.Cell(fill=Concrete, region=North_vault_wall_reg)
    Vault_floor_cell = openmc.Cell(fill=Concrete, region=Vault_floor_reg)
    Vault_ceiling_cell = openmc.Cell(fill=Concrete, region=Vault_ceiling_reg)
    West_vault_wall_cell = openmc.Cell(fill=Concrete, region=West_vault_wall_reg)
    East_vault_wall_cell = openmc.Cell(fill=Concrete, region=East_vault_wall_reg)
    South_vault_wall_cell = openmc.Cell(fill=Concrete, region=South_vault_wall_reg)
    foundation = openmc.Cell(fill=Concrete, region=Region_21)
    Vault_north_wall_ext_cell = openmc.Cell(
        fill=Concrete, region=Vault_north_wall_ext_reg
    )
    DANTE_vault_e_shield_cell = openmc.Cell(
        fill=IronConcrete, region=DANTE_vault_e_shield_reg
    )
    DANTE_vault_w_shield_cell = openmc.Cell(
        fill=IronConcrete, region=DANTE_vault_w_shield_reg
    )
    Vault_door_shield_s_pillar_cell = openmc.Cell(
        fill=RicoRad, region=Vault_door_shield_s_pillar_reg
    )
    Vault_door_shield_n_pillar_cell = openmc.Cell(
        fill=RicoRad, region=Vault_door_shield_n_pillar_reg
    )
    Vault_door_cell = openmc.Cell(fill=Concrete, region=Vault_door_reg)

    # Explicit declaration of the outer void
    Region_1000 = (
        -Surface_95
        & ~Vault_door_reg
        & ~Vault_door_shield_n_pillar_reg
        & ~Vault_door_shield_s_pillar_reg
        & ~DANTE_vault_w_shield_reg
        & ~DANTE_vault_e_shield_reg
        & ~Vault_north_wall_ext_reg
        & ~Region_21
        & ~South_vault_wall_reg
        & ~East_vault_wall_reg
        & ~West_vault_wall_reg
        & ~Vault_ceiling_reg
        & ~Vault_floor_reg
        & ~North_vault_wall_reg
        & ~I_beam_reg
        & ~DANTE_vault_bot_magnet_reg
        & ~DANTE_vault_top_magnet_reg
        & ~DANTE_vault_mag_stand_reg
        & ~Region_28
    )

    if overall_exclusion_region:
        Region_1000 = Region_1000 & ~overall_exclusion_region

    Cell_1000 = openmc.Cell(fill=Air, region=Region_1000)

    Cells = [
        Cell_1000,
        Vault_door_cell,
        Vault_door_shield_n_pillar_cell,
        Vault_door_shield_s_pillar_cell,
        DANTE_vault_w_shield_cell,
        DANTE_vault_e_shield_cell,
        Vault_north_wall_ext_cell,
        foundation,
        South_vault_wall_cell,
        East_vault_wall_cell,
        West_vault_wall_cell,
        Vault_ceiling_cell,
        Vault_floor_cell,
        North_vault_wall_cell,
        I_beam_cell,
        DANTE_vault_bot_magnet_cell,
        DANTE_vault_top_magnet_cell,
        DANTE_vault_mag_stand_cell,
        Vault_air_cell,
    ]

    Cells += added_cells

    Universe_1 = openmc.Universe(cells=Cells)
    geometry = openmc.Geometry(Universe_1)
    geometry.remove_redundant_surfaces()

    vault_model = openmc.model.Model(
        geometry=geometry, materials=materials, settings=settings, tallies=tallies
    )

    return vault_model


try:
    import openmc

    #
    # **** Natural elements ****
    #
    # Aluminum : 2.6989 g/cm3
    Aluminum = openmc.Material()
    Aluminum.set_density("g/cm3", 2.6989)
    Aluminum.add_nuclide("Al27", 1.0, "ao")

    # Copper : 8.96 g/cm3
    Material_2 = openmc.Material()
    Material_2.set_density("g/cm3", 8.96)
    Material_2.add_nuclide("Cu63", 0.6917, "ao")
    Material_2.add_nuclide("Cu65", 0.3083, "ao")

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

    # Name: Stainless steel 304
    # Density: 8.0 g/cm3
    # Reference: PNNL Report 15870 (Rev. 1)
    # Describes: vacuum pipes, flanges, general steel objects
    Material_6 = openmc.Material()
    Material_6.set_density("g/cm3", 8.0)
    Material_6.add_element("C", 0.00183, "ao")
    Material_6.add_element("Si", 0.009781, "ao")
    Material_6.add_nuclide("P31", 0.000408, "ao")
    Material_6.add_element("S", 0.000257, "ao")
    Material_6.add_element("Cr", 0.200762, "ao")
    Material_6.add_nuclide("Mn55", 0.010001, "ao")
    Material_6.add_element("Fe", 0.690375, "ao")
    Material_6.add_element("Ni", 0.086587, "ao")

    # Name: Wood (Southern Pine)
    # Density: 0.64 g/cm3
    # Reference: PNNL Report 15870 (Rev. 1)
    # Describes: doors
    Material_7 = openmc.Material()
    Material_7.set_density("g/cm3", 0.64)
    Material_7.add_nuclide("H1", 0.462423, "ao")
    Material_7.add_element("C", 0.323389, "ao")
    Material_7.add_nuclide("N14", 0.002773, "ao")
    Material_7.add_nuclide("O16", 0.208779, "ao")
    Material_7.add_element("Mg", 0.000639, "ao")
    Material_7.add_element("S", 0.001211, "ao")
    Material_7.add_element("K", 0.000397, "ao")
    Material_7.add_element("Ca", 0.000388, "ao")

    # Name: Gypsum (wallboard)
    # Density: 2.32 g/cm3
    # Reference: PNNL Report 15870 (Rev. 1)
    # Describes: drywall walls (GWB)
    Material_8 = openmc.Material()
    Material_8.set_density("g/cm3", 2.32)
    Material_8.add_nuclide("H1", 0.333321, "ao")
    Material_8.add_nuclide("O16", 0.500014, "ao")
    Material_8.add_element("S", 0.083324, "ao")
    Material_8.add_element("Ca", 0.083341, "ao")

    # **** Gamma shielding materials ****
    #
    # Tungsten : 19.3 g/cm3
    Material_10 = openmc.Material()
    Material_10.set_density("g/cm3", 19.3)
    Material_10.add_nuclide("W182", 0.265, "ao")
    Material_10.add_nuclide("W183", 0.1431, "ao")
    Material_10.add_nuclide("W184", 0.3064, "ao")
    Material_10.add_nuclide("W186", 0.2855, "ao")

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

    # Name: Non-borated polyethylene
    # Density: 0.93 g/cm3
    # Reference: PNNL Report 15870 (Rev. 1)
    # Describes: General purpose neutron shielding
    Polyethylene = openmc.Material()
    Polyethylene.set_density("g/cm3", 0.93)
    Polyethylene.add_nuclide("H1", 0.666662, "ao")
    Polyethylene.add_element("C", 0.333338, "ao")

    # High Density Polyethylene
    # Reference:  PNNL Report 15870 (Rev. 1)
    HDPE = openmc.Material(name="HDPE")
    HDPE.set_density("g/cm3", 0.95)
    HDPE.add_element("H", 0.143724, "wo")
    HDPE.add_element("C", 0.856276, "wo")

    # Name: Zirconium dihydride
    # Density: 5.6 g/cm3
    # Reference: JNM 386-388 (2009) 119-121
    # Describes: General purpose neutron shielding
    Material_22 = openmc.Material()
    Material_22.set_density("g/cm3", 5.6)
    Material_22.add_nuclide("H1", 0.0216, "wo")
    Material_22.add_element("Zr", 0.9784, "wo")

    # Name: Zirconium borohydride
    # Density: 1.18 g/cm3
    # Reference: JNM 386-388 (2009) 119-121
    # Describes: General purpose neutron shielding
    Material_23 = openmc.Material()
    Material_23.set_density("g/cm3", 1.18)
    Material_23.add_nuclide("H1", 0.1073, "wo")
    Material_23.add_nuclide("B10", 0.0571, "wo")
    Material_23.add_nuclide("B11", 0.23, "wo")
    Material_23.add_element("Zr", 0.6056, "wo")

    # Density: 1.848 g/cm3
    # Reference: None
    # Describes: Highest intenstiy neutron production target
    # Notes: Uses ENDF-derived proton nuclear data libray
    Material_30 = openmc.Material()
    Material_30.set_density("g/cm3", 1.848)
    Material_30.add_nuclide("Be9", 1.0, "ao")

    # Name: Concrete (Regular)
    # Density: 2.3 g/cm3
    # Reference: Provided by Matthey Carey, MIT EHS/RPP (mgcarey@mit.edu)
    # Describes: Facility walls, foundation, floors for activation calculations
    Material_40 = openmc.Material()
    Material_40.set_density("g/cm3", 2.3)
    Material_40.add_nuclide("Fe54", 2.0138e-05, "ao")
    Material_40.add_nuclide("Fe56", 0.00031874, "ao")
    Material_40.add_nuclide("Fe57", 7.2915e-06, "ao")
    Material_40.add_nuclide("Fe58", 1.0416e-06, "ao")
    Material_40.add_nuclide("H1", 0.01374, "ao")
    Material_40.add_nuclide("H2", 2.0613e-06, "ao")
    Material_40.add_nuclide("O16", 0.045685, "ao")
    Material_40.add_nuclide("O17", 1.8318e-05, "ao")
    Material_40.add_nuclide("Mg24", 9.0027e-05, "ao")
    Material_40.add_nuclide("Mg25", 1.1397e-05, "ao")
    Material_40.add_nuclide("Mg26", 1.2548e-05, "ao")
    Material_40.add_nuclide("Ca40", 0.001474, "ao")
    Material_40.add_nuclide("Ca42", 9.8378e-06, "ao")
    Material_40.add_nuclide("Ca43", 2.0527e-06, "ao")
    Material_40.add_nuclide("Ca44", 3.1718e-05, "ao")
    Material_40.add_nuclide("Ca46", 6.0821e-08, "ao")
    Material_40.add_nuclide("Ca48", 2.8434e-06, "ao")
    Material_40.add_nuclide("Si28", 0.015328, "ao")
    Material_40.add_nuclide("Si29", 0.00077613, "ao")
    Material_40.add_nuclide("Si30", 0.0005152, "ao")
    Material_40.add_nuclide("Na23", 0.00096395, "ao")
    Material_40.add_nuclide("K39", 0.00042949, "ao")
    Material_40.add_nuclide("K40", 4.6053e-08, "ao")
    Material_40.add_nuclide("K41", 3.0993e-05, "ao")
    Material_40.add_nuclide("Al27", 0.0017453, "ao")
    Material_40.add_nuclide("C12", 0.00011404, "ao")
    Material_40.add_nuclide("C13", 1.28e-06, "ao")

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

    # Previous model uses 10% borated high density polyethylene, but
    # according to Melhus, et. al., RicoRad consists of "2.00% mass boron
    # in a polyethylene-based matrix having a mass density of 0.945 g/cm^3"
    # Source:
    # Melhus, Christopher, et al. â€˜Storage Safe Shielding Assessment for a
    # HDR Californium-252 Brachytherapy Sourceâ€™.
    # Monte Carlo 2005 Topical Meeting, 01 2005, pp. 219â€“229.

    RicoRad = openmc.Material(name="RicoRad")
    RicoRad.set_density("g/cm3", 0.945)
    RicoRad.add_element("H", 0.14, percent_type="wo")
    RicoRad.add_element("C", 0.84, percent_type="wo")
    RicoRad.add_element("B", 0.02, percent_type="wo")

    ### LIBRA Materials
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

    Be = openmc.Material(name="Be")
    # Estimate Be temperature to be around 100 C
    # Be.temperature = 100 + 273
    Be.add_element("Be", 1.0, "ao")
    Be.set_density("g/cm3", 1.848)
except ImportError:
    pass
