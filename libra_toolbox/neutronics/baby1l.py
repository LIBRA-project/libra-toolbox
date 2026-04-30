import openmc
from libra_toolbox.neutronics.materials import *


def baby_geometry(x_c: float, y_c: float, z_c: float):
    """Returns the geometry for the BABY experiment.

    Args:
        x_c: x-coordinate of the center of the BABY experiment (cm)
        y_c: y-coordinate of the center of the BABY experiment (cm)
        z_c: z-coordinate of the center of the BABY experiment (cm)

    Returns:
        the sphere, cllif cell, and cells
    """

    epoxy_thickness = 2.54  # 1 inch
    alumina_compressed_thickness = 2.54  # 1 inch
    base_thickness = 0.786
    alumina_thickness = 0.635
    he_thickness = 0.6
    inconel_thickness = 0.3
    heater_gap = 0.878
    cllif_thickness = 6.388 + 0.13022  # without heater: 0.1081
    gap_thickness = 4.605
    cap = 1.422
    firebrick_thickness = 15.24
    high = 21.093
    cover = 2.392
    z_tab = 28.00
    lead_height = 4.00
    lead_width = 8.00
    lead_length = 16.00
    heater_r = 0.439
    heater_h = 25.40
    heater_z = (
        epoxy_thickness
        + alumina_compressed_thickness
        + base_thickness
        + alumina_thickness
        + he_thickness
        + inconel_thickness
        + heater_gap
        + z_c
    )

    cllif_radius = 7.00
    inconel_radius = 7.3
    he_radius = 9.144
    firebrick_radius = 12.002
    vessel_radius = 12.853
    external_radius = 13.272

    source_h = 50.00
    source_x = x_c - 13.50
    source_z = z_c - 5.635
    source_external_r = 5.00
    source_internal_r = 4.75

    ######## Surfaces #################
    z_plane_1 = openmc.ZPlane(0.0 + z_c)
    z_plane_2 = openmc.ZPlane(epoxy_thickness + z_c)
    z_plane_3 = openmc.ZPlane(epoxy_thickness + alumina_compressed_thickness + z_c)
    z_plane_4 = openmc.ZPlane(
        epoxy_thickness + alumina_compressed_thickness + base_thickness + z_c
    )
    z_plane_5 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + base_thickness
        + alumina_thickness
        + z_c
    )
    z_plane_6 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + base_thickness
        + alumina_thickness
        + he_thickness
        + z_c
    )
    z_plane_7 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + base_thickness
        + alumina_thickness
        + he_thickness
        + inconel_thickness
        + z_c
    )
    z_plane_8 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + base_thickness
        + alumina_thickness
        + he_thickness
        + inconel_thickness
        + cllif_thickness
        + z_c
    )
    z_plane_9 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + base_thickness
        + alumina_thickness
        + he_thickness
        + inconel_thickness
        + cllif_thickness
        + gap_thickness
        + z_c
    )
    z_plane_10 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + base_thickness
        + alumina_thickness
        + he_thickness
        + inconel_thickness
        + cllif_thickness
        + gap_thickness
        + cap
        + z_c
    )
    z_plane_11 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + base_thickness
        + alumina_thickness
        + firebrick_thickness
        + z_c
    )
    z_plane_12 = openmc.ZPlane(
        epoxy_thickness + alumina_compressed_thickness + base_thickness + high + z_c
    )
    z_plane_13 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + base_thickness
        + high
        + cover
        + z_c
    )
    z_plane_14 = openmc.ZPlane(z_c - z_tab)
    z_plane_15 = openmc.ZPlane(z_c - z_tab - epoxy_thickness)

    ######## Cylinder #################
    z_cyl_1 = openmc.ZCylinder(x0=x_c, y0=y_c, r=cllif_radius)
    z_cyl_2 = openmc.ZCylinder(x0=x_c, y0=y_c, r=inconel_radius)
    z_cyl_3 = openmc.ZCylinder(x0=x_c, y0=y_c, r=he_radius)
    z_cyl_4 = openmc.ZCylinder(x0=x_c, y0=y_c, r=firebrick_radius)
    z_cyl_5 = openmc.ZCylinder(x0=x_c, y0=y_c, r=vessel_radius)
    z_cyl_6 = openmc.ZCylinder(x0=x_c, y0=y_c, r=external_radius)

    right_cyl = openmc.model.RightCircularCylinder(
        (x_c, y_c, heater_z), heater_h, heater_r, axis="z"
    )
    ext_cyl_source = openmc.model.RightCircularCylinder(
        (source_x, y_c, source_z), source_h, source_external_r, axis="x"
    )
    source_region = openmc.model.RightCircularCylinder(
        (source_x + 0.25, y_c, source_z), source_h - 0.50, source_internal_r, axis="x"
    )

    ######## Sphere #################
    sphere = openmc.Sphere(x0=x_c, y0=y_c, z0=z_c, r=50.00)  # before r=50.00

    ######## Lead bricks positioned under the source #################
    lead_positions = [
        (x_c - 13.50, y_c, z_c - z_tab),
        (x_c - 2.96, y_c, z_c - z_tab),
        (x_c + 36.50, y_c, z_c - z_tab),
        (x_c + 25.96, y_c, z_c - z_tab),
    ]

    lead_blocks = []
    for position in lead_positions:
        lead_block_region = openmc.model.RectangularParallelepiped(
            position[0] - lead_width / 2,
            position[0] + lead_width / 2,
            position[1] - lead_length / 2,
            position[1] + lead_length / 2,
            position[2],
            position[2] + lead_height,
        )
        lead_blocks.append(lead_block_region)

    src_supp_length = 40.00
    src_supp_height = 20.00
    src_supp_width = 2.54
    src_supp_position = [
        (x_c - 13.50 + lead_width / 2, y_c, z_c - z_tab),
        (x_c + 25.96 + lead_width / 2, y_c, z_c - z_tab),
    ]
    src_supports = []
    for position in src_supp_position:
        source_support = openmc.model.RectangularParallelepiped(
            position[0],
            position[0] + src_supp_width,
            position[1] - src_supp_length / 2,
            position[1] + src_supp_length / 2,
            position[2],
            position[2] + src_supp_height,
        )
        src_supports.append(source_support)

    # regions
    source_wall_region = -ext_cyl_source & +source_region
    source_region = -source_region
    epoxy_region = +z_plane_1 & -z_plane_2 & -sphere
    alumina_compressed_region = +z_plane_2 & -z_plane_3 & -sphere
    bottom_vessel = +z_plane_3 & -z_plane_4 & -z_cyl_6
    top_vessel = +z_plane_12 & -z_plane_13 & -z_cyl_6 & +right_cyl
    cylinder_vessel = +z_plane_4 & -z_plane_12 & +z_cyl_5 & -z_cyl_6
    vessel_region = bottom_vessel | cylinder_vessel | top_vessel
    alumina_region = +z_plane_4 & -z_plane_5 & -z_cyl_5
    bottom_cap = +z_plane_6 & -z_plane_7 & -z_cyl_2 & +right_cyl
    cylinder_cap = +z_plane_7 & -z_plane_9 & +z_cyl_1 & -z_cyl_2 & +right_cyl
    top_cap = +z_plane_9 & -z_plane_10 & -z_cyl_2 & +right_cyl
    cap_region = bottom_cap | cylinder_cap | top_cap
    cllif_region = +z_plane_7 & -z_plane_8 & -z_cyl_1 & +right_cyl
    gap_region = +z_plane_8 & -z_plane_9 & -z_cyl_1 & +right_cyl
    firebrick_region = +z_plane_5 & -z_plane_11 & +z_cyl_3 & -z_cyl_4
    heater_region = -right_cyl
    table_under_source_region = +z_plane_15 & -z_plane_14 & -sphere
    lead_block_1_region = -lead_blocks[0]
    lead_block_2_region = -lead_blocks[1]
    lead_block_3_region = -lead_blocks[2]
    lead_block_4_region = -lead_blocks[3]
    src_supp_1_region = -src_supports[0] & ~source_wall_region & ~source_region
    src_supp_2_region = -src_supports[1] & ~source_wall_region & ~source_region
    he_region = (
        +z_plane_5
        & -z_plane_12
        & -z_cyl_5
        & ~source_region
        & ~epoxy_region
        & ~alumina_compressed_region
        & ~alumina_region
        & ~cllif_region
        & ~gap_region
        & ~firebrick_region
        & ~vessel_region
        & ~cap_region
        & ~heater_region
        & ~table_under_source_region
        & ~lead_block_1_region
        & ~lead_block_2_region
        & ~lead_block_3_region
        & ~lead_block_4_region
        & ~src_supp_1_region
        & ~src_supp_2_region
    )
    sphere_region = (
        -sphere
        & ~source_wall_region
        & ~source_region
        & ~epoxy_region
        & ~alumina_compressed_region
        & ~alumina_region
        & ~cllif_region
        & ~gap_region
        & ~firebrick_region
        & ~he_region
        & ~vessel_region
        & ~cap_region
        & ~heater_region
        & ~table_under_source_region
        & ~lead_block_1_region
        & ~lead_block_2_region
        & ~lead_block_3_region
        & ~lead_block_4_region
        & ~src_supp_1_region
        & ~src_supp_2_region
    )

    # cells
    source_wall_cell_1 = openmc.Cell(region=source_wall_region)
    source_wall_cell_1.fill = SS304
    source_region = openmc.Cell(region=source_region)
    source_region.fill = None
    epoxy_cell = openmc.Cell(region=epoxy_region)
    epoxy_cell.fill = Epoxy
    alumina_compressed_cell = openmc.Cell(region=alumina_compressed_region)
    alumina_compressed_cell.fill = Alumina
    vessel_cell = openmc.Cell(region=vessel_region)
    vessel_cell.fill = Inconel625
    alumina_cell = openmc.Cell(region=alumina_region)
    alumina_cell.fill = Alumina
    cllif_cell = openmc.Cell(region=cllif_region)
    cllif_cell.fill = Cllif  # Cllif or lithium_lead
    gap_cell = openmc.Cell(region=gap_region)
    gap_cell.fill = Helium
    cap_cell = openmc.Cell(region=cap_region)
    cap_cell.fill = Inconel625
    firebrick_cell = openmc.Cell(region=firebrick_region)
    firebrick_cell.fill = Firebrick
    heater_cell = openmc.Cell(region=heater_region)
    heater_cell.fill = Heater_mat
    table_cell = openmc.Cell(region=table_under_source_region)
    table_cell.fill = Epoxy
    sphere_cell = openmc.Cell(region=sphere_region)
    sphere_cell.fill = Air
    he_cell = openmc.Cell(region=he_region)
    he_cell.fill = Helium
    lead_block_1_cell = openmc.Cell(region=lead_block_1_region)
    lead_block_1_cell.fill = Lead
    lead_block_2_cell = openmc.Cell(region=lead_block_2_region)
    lead_block_2_cell.fill = Lead
    lead_block_3_cell = openmc.Cell(region=lead_block_3_region)
    lead_block_3_cell.fill = Lead
    lead_block_4_cell = openmc.Cell(region=lead_block_4_region)
    lead_block_4_cell.fill = Lead
    src_supp_1_cell = openmc.Cell(region=src_supp_1_region)
    src_supp_1_cell.fill = HDPE
    src_supp_2_cell = openmc.Cell(region=src_supp_2_region)
    src_supp_2_cell.fill = HDPE

    cells = [
        source_wall_cell_1,
        source_region,
        epoxy_cell,
        alumina_compressed_cell,
        vessel_cell,
        alumina_cell,
        cap_cell,
        cllif_cell,
        gap_cell,
        firebrick_cell,
        heater_cell,
        he_cell,
        sphere_cell,
        table_cell,
        lead_block_1_cell,
        lead_block_2_cell,
        lead_block_3_cell,
        lead_block_4_cell,
        src_supp_1_cell,
        src_supp_2_cell,
    ]

    return sphere, cllif_cell, cells

