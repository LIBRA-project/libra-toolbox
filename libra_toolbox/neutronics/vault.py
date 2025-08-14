def build_vault_model(
    settings=None,
    tallies=None,
    added_cells=[],
    added_materials=[],
    overall_exclusion_region=None,
    cross_sections_destination="cross_sections",
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
        import openmc_data_downloader as odd
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "openmc and openmc_data_downloader are required.")

    from .materials import Aluminum, Air, Concrete, IronConcrete, RicoRad, SS304, Copper

    materials = openmc.Materials(
        [
            Aluminum,
            Air,
            Concrete,
            IronConcrete,
            RicoRad,
            SS304,
            Copper,
        ]
    )

    # Add materials from imported model
    for mat in added_materials:
        if mat not in materials:
            materials.append(mat)

    materials.download_cross_section_data(
        libraries=["ENDFB-8.0-NNDC"],
        set_OPENMC_CROSS_SECTIONS=True,
        particles=["neutron"],
        destination=cross_sections_destination,
    )
    #
    # Definition of the spherical void/blackhole boundary
    Surface_95 = openmc.Sphere(
        x0=0.0, y0=0.0, z0=0.0, r=2500.0, boundary_type="vacuum")

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
    Surface_48 = openmc.Plane(a=0.99881, b=-0.04885,
                              c=0.0, d=964.9095439999999)

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
        fill=Copper, region=DANTE_vault_top_magnet_reg
    )
    DANTE_vault_bot_magnet_cell = openmc.Cell(
        fill=Copper, region=DANTE_vault_bot_magnet_reg
    )
    I_beam_cell = openmc.Cell(fill=SS304, region=I_beam_reg)
    North_vault_wall_cell = openmc.Cell(
        fill=Concrete, region=North_vault_wall_reg)
    Vault_floor_cell = openmc.Cell(fill=Concrete, region=Vault_floor_reg)
    Vault_ceiling_cell = openmc.Cell(fill=Concrete, region=Vault_ceiling_reg)
    West_vault_wall_cell = openmc.Cell(
        fill=Concrete, region=West_vault_wall_reg)
    East_vault_wall_cell = openmc.Cell(
        fill=Concrete, region=East_vault_wall_reg)
    South_vault_wall_cell = openmc.Cell(
        fill=Concrete, region=South_vault_wall_reg)
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
