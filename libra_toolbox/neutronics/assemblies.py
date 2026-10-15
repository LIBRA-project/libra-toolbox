from dataclasses import dataclass, field
from .components import *
import openmc


@dataclass
class Placement:
    """A component or nested assembly placed relative to a parent assembly."""

    item: object
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Assembly:
    """A collection of components and nested assemblies.

    Positions are expressed in the parent assembly's coordinate system.
    Rotations are Euler angles in degrees about the local x, y, and z axes.
    Each child's local origin is placed at the specified position.
    """

    name: str
    placements: list[Placement] = field(default_factory=list)

    def add(
        self,
        item: object,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Placement:
        """Add a component or nested assembly."""

        if not isinstance(item, (Component, Assembly)):
            raise TypeError(
                "An assembly can contain only Component or Assembly objects."
            )

        self._validate_vector(position, "position")
        self._validate_vector(rotation, "rotation")

        placement = Placement(
            item=item,
            position=tuple(position),
            rotation=tuple(rotation),
        )
        self.placements.append(placement)
        return placement

    @staticmethod
    def _validate_vector(value, label: str) -> None:
        if len(value) != 3:
            raise ValueError(f"{label} must contain exactly three values.")

    def describe(self, indent: int = 0) -> str:
        """Return a readable description of this assembly and its contents."""

        prefix = " " * indent
        lines = [f"{prefix}Assembly: {self.name}"]

        if not self.placements:
            lines.append(f"{prefix}  (empty)")
            return "\n".join(lines)

        for placement in self.placements:
            item = placement.item
            item_prefix = prefix + "  "

            lines.append(f"{item_prefix}{item.name}")
            lines.append(f"{item_prefix}  Position: {placement.position}")
            lines.append(f"{item_prefix}  Rotation: {placement.rotation}")

            if isinstance(item, Assembly):
                lines.append(item.describe(indent=indent + 4))
            else:
                lines.append(
                    f"{item_prefix}  Reference point: "
                    f"{item.reference_point}"
                )

        return "\n".join(lines)
    

@dataclass
class BreederVessel1LAssembly:
    name: str
    vessel: Vessel1L
    breeder: Breeder
    headspace: HeadSpace

    def geometry(self):
        # Vessel structure
        vessel_cells = self.vessel.geometry()

        # Breeder region inside the vessel
        breeder_region = self.breeder.region(self.vessel)

        breeder_cell = openmc.Cell(
            name=self.breeder.name,
            region=breeder_region,
            fill=self.breeder.material,
        )

        # Everything inside the vessel not occupied by breeder
        headspace_region = self.vessel.fill_region & ~breeder_region

        headspace_cell = openmc.Cell(
            name=self.headspace.name,
            region=headspace_region,
            fill=self.headspace.material,
        )

        return vessel_cells + [
            breeder_cell,
            headspace_cell,
        ]