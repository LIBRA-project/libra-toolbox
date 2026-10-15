from dataclasses import dataclass
from typing import Tuple, Optional
import openmc
import materials
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from materials import macor


@dataclass(kw_only=True)
class Component:
    """Base experiment component.

    Position and rotation are expressed in the experiment coordinate system.
    Measures are always in (cm).
    """

    name: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)


    @property
    def reference_point(self) -> str:
        return "Not specified"

    @property
    def local_axes_description(self) -> str:
        return "Not specified"

    def describe(self) -> str:
        """Return a human-readable description of this component."""
        return (
            f"Component: {self.name}\n"
            f"Reference point: {self.reference_point}\n"
            f"Local origin: (0, 0, 0)\n"
            f"Local axes: {self.local_axes_description}\n"
            f"Geometry: Defined in local coordinates"
        )


@dataclass
class LeadBrick(Component):
    width: float = 8.0
    length: float = 16.0
    height: float = 4.0
    material: openmc.Material = materials.Lead

    @property
    def reference_point(self):
        return "Geometric center of the cuboid brick"

    def geometry(self):
        box = openmc.model.RectangularParallelepiped(
            -self.width / 2,
            +self.width / 2,
            -self.length / 2,
            +self.length / 2,
            0,
            self.height,
        )

        cell = openmc.Cell(
            name=self.name,
            region=-box,
            fill=self.material,
        )

        return [cell]

@dataclass
class GeneratorSupport(Component):
    """Support structure for the neutron generator."""

    width: float = 2.54
    length: float = 2.54
    height: float = 2.54

    separation: float = 10.0

    material: openmc.Material = materials.HDPE

    def geometry(self):
        """Build the two generator support blocks."""

        cells = []

        for i, y in enumerate(
            (-self.separation / 2, self.separation / 2),
            start=1,
        ):
            box = openmc.model.RectangularParallelepiped(
                -self.width / 2,
                +self.width / 2,
                y - self.length / 2,
                y + self.length / 2,
                0.0,
                self.height,
            )

            cells.append(
                openmc.Cell(
                    name=f"{self.name}_{i}",
                    region=-box,
                    fill=self.material,
                )
            )

        return cells

@dataclass
class InsulatorLagging(Component):
    radius: float = 12.853
    thickness: float = 0.635
    material: openmc.Material = materials.Alumina

    def geometry(self):
        bottom = openmc.ZPlane(z0=0.0)
        top = openmc.ZPlane(z0=self.thickness)
        outer = openmc.ZCylinder(r=self.radius)

        region = +bottom & -top & -outer

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


@dataclass
class Furnace(Component):
    inner_radius: float = 9.144
    outer_radius: float = 12.002
    thickness: float = 15.24
    material: openmc.Material = materials.Firebrick

    def geometry(self):
        bottom = openmc.ZPlane(z0=0.0)
        top = openmc.ZPlane(z0=self.thickness)

        inner = openmc.ZCylinder(r=self.inner_radius)
        outer = openmc.ZCylinder(r=self.outer_radius)

        region = (
            +bottom
            & -top
            & +inner
            & -outer
        )

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


@dataclass
class InsulatorSlab(Component):
    radius: float = 50.0
    thickness: float = 2.54
    material: openmc.Material = materials.Epoxy

    def geometry(self):
        bottom = openmc.ZPlane(z0=0.0)
        top = openmc.ZPlane(z0=self.thickness)
        outer = openmc.ZCylinder(r=self.radius)

        region = +bottom & -top & -outer

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


@dataclass
class Heater(Component):
    radius: float = 0.439
    height: float = 25.40
    material: openmc.Material = materials.Heater_mat

    def geometry(self):
        cylinder = openmc.ZCylinder(r=self.radius)

        bottom = openmc.ZPlane(z0=0.0)
        top = openmc.ZPlane(z0=self.height)

        region = +bottom & -top & -cylinder

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


@dataclass
class Table(Component):
    width: float = 100.0
    length: float = 100.0
    thickness: float = 2.54
    material: openmc.Material = materials.Epoxy

    def geometry(self):
        x_min = -self.width / 2
        x_max = self.width / 2
        y_min = -self.length / 2
        y_max = self.length / 2

        bottom = openmc.ZPlane(z0=0.0)
        top = openmc.ZPlane(z0=self.thickness)

        region = (
            +bottom
            & -top
            & +openmc.XPlane(x0=x_min)
            & -openmc.XPlane(x0=x_max)
            & +openmc.YPlane(y0=y_min)
            & -openmc.YPlane(y0=y_max)
        )

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]


@dataclass
class Vessel1L(Component):
    # Main vessel dimensions (cm)
    radius: float = 7.0
    external_radius: float = 7.3
    base_thickness: float = 0.2
    height: float = 11.14
    cover_thickness: float = 0.5

    # Central socket dimensions (cm)
    socket_radius: float = 0.95
    socket_external_radius: float = 1.1
    socket_depth: float = 10.65
    socket_base_thickness: float = 0.15

    # Main vessel material
    material: openmc.Material = None

    # Cap thickness for the three large pipes (cm)
    pipe_cap_thickness: float = 0.3

    # All three bolt sleeves are raised by this offset (cm)
    bolt_height_offset: float = 0.6

    @property
    def reference_point(self):
        return "Center of the vessel bottom's bottom cylindricalsurface (z=0)."

    @property
    def local_axes_description(self) -> str:
        return "+z points from the bottom toward the lid"

    # ------------------------------------------------------------------
    # Derived elevations
    # ------------------------------------------------------------------

    @property
    def fill_bottom(self):
        return self.base_thickness

    @property
    def fill_top(self):
        return self.fill_bottom + self.height

    @property
    def lid_top(self):
        return self.fill_top + self.cover_thickness

    @property
    def socket_base_top(self):
        return self.lid_top - self.socket_depth

    @property
    def socket_base_bottom(self):
        return self.socket_base_top - self.socket_base_thickness

    @property
    def pipe_penetration_bottom(self):
        # Large pipes extend 0.8 cm below the original lid.
        return self.fill_top - 0.8

    # ------------------------------------------------------------------
    # Three existing large pipes and bolt sleeves
    # ------------------------------------------------------------------

    @property
    def pipe_data(self):
        """
        Each entry:
        name, x, y, inner radius, outer radius, pipe height above lid,
        bolt offset above lid, bolt height, bolt outer radius.
        """
        return [
            (
                "TwinPipe1",
                -5.7, 0.0,
                0.50, 0.65,
                7.6,
                2.0, 3.5, 1.5,
            ),
            (
                "TwinPipe2",
                5.7, 0.0,
                0.50, 0.65,
                7.6,
                2.0, 3.5, 1.5,
            ),
            (
                "ThirdPipe",
                0.0, 5.85,
                0.85, 0.95,
                4.85,
                0.5, 3.3, 1.9,
            ),
        ]

    # ------------------------------------------------------------------
    # Five additional small pipes
    # ------------------------------------------------------------------

    @property
    def small_pipe_data(self):
        """
        Each entry: name, x, y, is_internal.

        One internal pipe at (0, -5.7).
        Four external pipes tangent to the outer vessel radius,
        positioned at 30, 60, 120 and 150 degrees on the -y side.
        """
        pipes = [
            ("SmallPipeInternal", 0.0, -5.7, True),
        ]

        center_radius = self.external_radius + 0.32

        for angle_deg in (30, 60, 120, 150):
            angle = np.deg2rad(angle_deg)
            x = center_radius * np.cos(angle)
            y = -center_radius * np.sin(angle)

            pipes.append(
                (
                    f"SmallPipeExternal{angle_deg}",
                    x,
                    y,
                    False,
                )
            )

        return pipes

    @property
    def small_pipe_bottom(self):
        return self.lid_top - 8.4

    @property
    def small_pipe_cap_thickness(self):
        return 0.15

    # ------------------------------------------------------------------
    # Internal fill region
    # ------------------------------------------------------------------

    @property
    def fill_region(self):
        """
        Region available to breeder/headspace inside the vessel.

        Excludes the socket, the three large pipes, and the internal
        small pipe. External small pipes do not remove internal volume.
        """
        vessel_inner = openmc.ZCylinder(r=self.radius)
        bottom = openmc.ZPlane(z0=self.fill_bottom)
        top = openmc.ZPlane(z0=self.fill_top)

        region = -vessel_inner & +bottom & -top

        # Central socket footprint, including its base and wall.
        socket_outer = openmc.ZCylinder(
            r=self.socket_external_radius
        )
        socket_bottom = openmc.ZPlane(
            z0=self.socket_base_bottom
        )

        region &= ~(
            -socket_outer
            & +socket_bottom
            & -top
        )

        # Three large pipe footprints below the original lid.
        large_pipe_bottom = openmc.ZPlane(
            z0=self.pipe_penetration_bottom
        )

        for (
            name, x, y, inner_r, outer_r, pipe_height,
            bolt_offset, bolt_height, bolt_outer_r
        ) in self.pipe_data:
            pipe_outer = openmc.ZCylinder(
                x0=x, y0=y, r=outer_r
            )

            region &= ~(
                -pipe_outer
                & +large_pipe_bottom
                & -top
            )

        # Internal small pipe footprint.
        small_outer = openmc.ZCylinder(
            x0=0.0, y0=-5.7, r=0.32
        )
        small_bottom = openmc.ZPlane(
            z0=self.small_pipe_bottom
        )

        region &= ~(
            -small_outer
            & +small_bottom
            & -top
        )

        return region

    # ------------------------------------------------------------------
    # Available fill area and volume
    # ------------------------------------------------------------------

    def available_area(self, z):
        if z < self.fill_bottom or z > self.fill_top:
            return 0.0

        area = np.pi * self.radius**2

        if self.socket_base_bottom <= z <= self.fill_top:
            area -= np.pi * self.socket_external_radius**2

        if self.pipe_penetration_bottom <= z <= self.fill_top:
            for (
                name, x, y, inner_r, outer_r, pipe_height,
                bolt_offset, bolt_height, bolt_outer_r
            ) in self.pipe_data:
                area -= np.pi * outer_r**2

        if self.small_pipe_bottom <= z <= self.fill_top:
            area -= np.pi * 0.32**2

        return max(area, 0.0)

    def volume_below(self, z):
        """Available fill volume between fill_bottom and elevation z."""
        z = min(max(z, self.fill_bottom), self.fill_top)

        breakpoints = [
            point
            for point in (
                self.socket_base_bottom,
                self.pipe_penetration_bottom,
                self.small_pipe_bottom,
            )
            if self.fill_bottom < point < z
        ]

        volume, _ = quad(
            self.available_area,
            self.fill_bottom,
            z,
            points=breakpoints or None,
        )

        return volume

    def height_for_volume(self, volume):
        """Return fill height above fill_bottom for a volume in cm³."""
        if volume < 0:
            raise ValueError("Fill volume cannot be negative.")

        max_volume = self.volume_below(self.fill_top)

        if volume > max_volume:
            raise ValueError(
                f"Requested volume ({volume:.6g} cm³) exceeds "
                f"available volume ({max_volume:.6g} cm³)."
            )

        if volume == 0:
            return 0.0

        return brentq(
            lambda h: (
                self.volume_below(self.fill_bottom + h) - volume
            ),
            0.0,
            self.height,
        )

    # ------------------------------------------------------------------
    # OpenMC geometry
    # ------------------------------------------------------------------

    def geometry(self):
        if self.material is None:
            raise ValueError("Vessel1L requires a vessel material.")

        cells = []

        # Common surfaces
        inner_cyl = openmc.ZCylinder(r=self.radius)
        outer_cyl = openmc.ZCylinder(r=self.external_radius)

        base_bottom = openmc.ZPlane(z0=0.0)
        base_top = openmc.ZPlane(z0=self.fill_bottom)
        fill_top = openmc.ZPlane(z0=self.fill_top)
        lid_top = openmc.ZPlane(z0=self.lid_top)

        # --------------------------------------------------------------
        # 1. Vessel bottom
        # --------------------------------------------------------------

        bottom_region = (
            -outer_cyl
            & +base_bottom
            & -base_top
        )

        cells.append(
            openmc.Cell(
                name=f"{self.name}_Bottom",
                fill=self.material,
                region=bottom_region,
            )
        )

        # --------------------------------------------------------------
        # 2. Cylindrical vessel wall
        # --------------------------------------------------------------

        wall_region = (
            +inner_cyl
            & -outer_cyl
            & +base_top
            & -fill_top
        )

        cells.append(
            openmc.Cell(
                name=f"{self.name}_Wall",
                fill=self.material,
                region=wall_region,
            )
        )

        # --------------------------------------------------------------
        # 3. Original vessel lid
        # --------------------------------------------------------------

        socket_outer = openmc.ZCylinder(
            r=self.socket_external_radius
        )

        lid_region = (
            -outer_cyl
            & +fill_top
            & -lid_top
        )

        # Socket opening
        lid_region &= ~(-socket_outer)

        # Openings for the three large pipes
        for (
            name, x, y, inner_r, outer_r, pipe_height,
            bolt_offset, bolt_height, bolt_outer_r
        ) in self.pipe_data:
            pipe_outer = openmc.ZCylinder(
                x0=x, y0=y, r=outer_r
            )
            lid_region &= ~(-pipe_outer)

        # Opening for the internal small pipe
        small_internal_outer = openmc.ZCylinder(
            x0=0.0, y0=-5.7, r=0.32
        )
        lid_region &= ~(-small_internal_outer)

        cells.append(
            openmc.Cell(
                name=f"{self.name}_Lid",
                fill=self.material,
                region=lid_region,
            )
        )

        # --------------------------------------------------------------
        # 4. Central socket base
        # --------------------------------------------------------------

        socket_base_bottom = openmc.ZPlane(
            z0=self.socket_base_bottom
        )
        socket_base_top = openmc.ZPlane(
            z0=self.socket_base_top
        )

        socket_base_region = (
            -socket_outer
            & +socket_base_bottom
            & -socket_base_top
        )

        cells.append(
            openmc.Cell(
                name=f"{self.name}_SocketBase",
                fill=self.material,
                region=socket_base_region,
            )
        )

        # --------------------------------------------------------------
        # 5. Central socket wall
        # --------------------------------------------------------------

        socket_inner = openmc.ZCylinder(
            r=self.socket_radius
        )

        socket_wall_region = (
            +socket_inner
            & -socket_outer
            & +socket_base_top
            & -lid_top
        )

        cells.append(
            openmc.Cell(
                name=f"{self.name}_SocketWall",
                fill=self.material,
                region=socket_wall_region,
            )
        )

        # --------------------------------------------------------------
        # 6. Three large pipes, bolt sleeves, and solid caps
        # --------------------------------------------------------------

        large_pipe_bottom = openmc.ZPlane(
            z0=self.pipe_penetration_bottom
        )

        for (
            name, x, y, inner_r, outer_r, pipe_height,
            bolt_offset, bolt_height, bolt_outer_r
        ) in self.pipe_data:

            pipe_inner = openmc.ZCylinder(
                x0=x, y0=y, r=inner_r
            )
            pipe_outer = openmc.ZCylinder(
                x0=x, y0=y, r=outer_r
            )

            pipe_top_z = self.lid_top + pipe_height
            pipe_wall_top_z = pipe_top_z - self.pipe_cap_thickness

            pipe_wall_top = openmc.ZPlane(
                z0=pipe_wall_top_z
            )

            # Raise each bolt sleeve by 0.6 cm.
            bolt_bottom_z = (
                self.lid_top
                + bolt_offset
                + self.bolt_height_offset
            )
            bolt_top_z = bolt_bottom_z + bolt_height

            bolt_bottom = openmc.ZPlane(z0=bolt_bottom_z)
            bolt_top = openmc.ZPlane(z0=bolt_top_z)

            # Hollow bolt sleeve, matching the pipe's inner radius.
            bolt_inner = openmc.ZCylinder(
                x0=x, y0=y, r=inner_r
            )
            bolt_outer_surface = openmc.ZCylinder(
                x0=x, y0=y, r=bolt_outer_r
            )

            bolt_region = (
                +bolt_inner
                & -bolt_outer_surface
                & +bolt_bottom
                & -bolt_top
            )

            # Pipe wall, excluding the bolt sleeve where they overlap.
            pipe_wall_region = (
                +pipe_inner
                & -pipe_outer
                & +large_pipe_bottom
                & -pipe_wall_top
            )
            pipe_wall_region &= ~bolt_region

            cells.append(
                openmc.Cell(
                    name=f"{self.name}_{name}",
                    fill=self.material,
                    region=pipe_wall_region,
                )
            )

            cells.append(
                openmc.Cell(
                    name=f"{self.name}_{name}_Bolt",
                    fill=self.material,
                    region=bolt_region,
                )
            )

            # The pipe bores remain void.

            # Solid 0.3 cm cap at the top of each pipe.
            cap_bottom = openmc.ZPlane(
                z0=pipe_wall_top_z
            )
            cap_top = openmc.ZPlane(
                z0=pipe_top_z
            )

            cap_region = (
                -pipe_outer
                & +cap_bottom
                & -cap_top
            )

            cells.append(
                openmc.Cell(
                    name=f"{self.name}_{name}_Cap",
                    fill=self.material,
                    region=cap_region,
                )
            )

        # --------------------------------------------------------------
        # 7. Five small pipes and bottom caps
        # --------------------------------------------------------------

        small_bottom_z = self.small_pipe_bottom
        small_cap_top_z = (
            small_bottom_z + self.small_pipe_cap_thickness
        )

        small_bottom = openmc.ZPlane(z0=small_bottom_z)
        small_cap_top = openmc.ZPlane(z0=small_cap_top_z)

        for name, x, y, is_internal in self.small_pipe_data:
            small_inner = openmc.ZCylinder(
                x0=x, y0=y, r=0.18
            )
            small_outer = openmc.ZCylinder(
                x0=x, y0=y, r=0.32
            )

            small_wall_region = (
                +small_inner
                & -small_outer
                & +small_cap_top
                & -lid_top
            )

            cells.append(
                openmc.Cell(
                    name=f"{self.name}_{name}",
                    fill=self.material,
                    region=small_wall_region,
                )
            )

            small_cap_region = (
                -small_outer
                & +small_bottom
                & -small_cap_top
            )

            cells.append(
                openmc.Cell(
                    name=f"{self.name}_{name}_Cap",
                    fill=self.material,
                    region=small_cap_region,
                )
            )

            # No cell is created inside the small-pipe bore.

        # --------------------------------------------------------------
        # 8. Additional 0.6 cm lid
        # --------------------------------------------------------------

        # Additional 0.6 cm lid, with the same openings as the original lid
        additional_lid_top_z = self.lid_top + 0.6
        additional_lid_top = openmc.ZPlane(z0=additional_lid_top_z)

        additional_lid_material = macor

        additional_lid_region = (
            -outer_cyl
            & +lid_top
            & -additional_lid_top
        )

        # Opening for the central socket
        additional_lid_region &= ~(-socket_outer)

        # Openings for the three large pipes
        for (
            name, x, y, inner_r, outer_r, pipe_height,
            bolt_offset, bolt_height, bolt_outer_r
        ) in self.pipe_data:
            pipe_outer = openmc.ZCylinder(x0=x, y0=y, r=outer_r)
            additional_lid_region &= ~(-pipe_outer)

        cells.append(
            openmc.Cell(
                name=f"{self.name}_AdditionalLid",
                fill=additional_lid_material,
                region=additional_lid_region,
            )
        )

        return cells
    
@dataclass
class Breeder(Component):
    material: openmc.Material

    density: float | None = None
    mass: float | None = None
    volume: float | None = None

    @property
    def requested_volume(self):
        if self.volume is not None:
            return self.volume

        if self.mass is not None:
            if self.density is None:
                raise ValueError(
                    "Density is required when mass is specified."
                )
            return self.mass / self.density

        raise ValueError(
            "Specify either volume or mass."
        )

    def region(self, vessel):
        fill_height = vessel.height_for_volume(
            self.requested_volume
        )

        bottom = openmc.ZPlane(z0=vessel.fill_bottom)
        top = openmc.ZPlane(z0=vessel.fill_bottom + fill_height)

        return vessel.fill_region & +bottom & -top

@dataclass
class HeadSpace:
    name: str
    material: openmc.Material


@dataclass
class OuterVessel(Component):
    """Outer cylindrical vessel wall."""

    inner_radius: float = 12.853
    outer_radius: float = 13.272
    height: float = 21.093

    bottom_thickness: float = 0.786
    top_thickness: float = 2.392

    material: openmc.Material = None

    def geometry(self):
        # Main cylindrical wall
        inner = openmc.ZCylinder(r=self.inner_radius)
        outer = openmc.ZCylinder(r=self.outer_radius)

        bottom = openmc.ZPlane(z0=0.0)
        wall_bottom = openmc.ZPlane(z0=self.bottom_thickness)
        wall_top = openmc.ZPlane(
            z0=self.bottom_thickness + self.height
        )
        top = openmc.ZPlane(
            z0=self.bottom_thickness
            + self.height
            + self.top_thickness
        )

        wall_region = (
            +wall_bottom
            & -wall_top
            & +inner
            & -outer
        )

        bottom_region = (
            +bottom
            & -wall_bottom
            & -outer
        )

        top_region = (
            +wall_top
            & -top
            & -outer
        )

        region = wall_region | bottom_region | top_region

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]

@dataclass
class OuterVesselSweepGas(Component):
    """Sweep gas filling the region inside the outer vessel but
    outside the inner Vessel1L and Furnace.

    Parameters
    ----------
    material : openmc.Material
        Material filling the sweep-gas region.
    """

    material: openmc.Material

    def region(
        self,
        outer_vessel,
        vessel_1l,
        furnace=None,
    ):
        """Return the sweep-gas region."""

        # Start with the entire internal volume of the outer vessel.
        region = outer_vessel.fill_region

        # Exclude the inner Vessel1L.
        region &= ~vessel_1l.region

        # Exclude the Furnace, if present.
        if furnace is not None:
            region &= ~furnace.region

        return region

    def geometry(
        self,
        outer_vessel,
        vessel_1l,
        furnace=None,
    ):
        """Build the OpenMC sweep-gas cell."""

        region = self.region(
            outer_vessel=outer_vessel,
            vessel_1l=vessel_1l,
            furnace=furnace,
        )

        cell = openmc.Cell(
            name=self.name,
            region=region,
            fill=self.material,
        )

        return [cell]
