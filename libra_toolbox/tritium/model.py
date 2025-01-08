import pint
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import solve_ivp

from libra_toolbox.tritium import ureg

from typing import Callable

SPECIFIC_ACT = 3.57e14 * ureg.Bq * ureg.g**-1
MOLAR_MASS = 6.032 / 2 * ureg.g * ureg.mol**-1


class Model:
    r"""
    0D model for tritium release from a cylindrical salt blanket

    The model is a simple ODE:

    .. math::
        V \frac{d c_\mathrm{salt}}{dt} = S - Q_\mathrm{wall} - Q_\mathrm{top}

    where :math:`V` is the volume of the salt, :math:`c_\mathrm{salt}` is the tritium concentration in the salt, :math:`S` is the tritium source term, and :math:`Q_i` are the different release rates.

    The source term is expressed as:

    .. math::
        S = \mathrm{TBR} \cdot \Gamma_n

    where :math:`\mathrm{TBR}` is the Tritium Breeding Ratio and :math:`\Gamma_n` is the neutron rate.

    The release rates are expressed as:

    .. math::
        Q_i = A_i \ k_i \ (c_\mathrm{salt} - c_\mathrm{external}) \approx A_i \ k_i \ c_\mathrm{salt}

    where :math:`A_i` is the surface area, :math:`k_i` is the mass transport coefficient, and :math:`c_\mathrm{external}` is the external tritium concentration (assumed negligible compared to :math:`c_\mathrm{salt}`).

    Args:
        radius (pint.Quantity): radius of the salt
        height (pint.Quantity): height of the salt
        TBR (pint.Quantity): Tritium Breeding Ratio
        neutron_rate (pint.Quantity): neutron rate
        k_wall (pint.Quantity): mass transport coefficient for the
            walls
        k_top (pint.Quantity): mass transport coefficient for the
            top surface
        irradiations (list[tuple[pint.Quantity, pint.Quantity]]): list
            of tuples with the start and stop times of irradiations

    Attributes:
        radius (pint.Quantity): radius of the salt
        height (pint.Quantity): height of the salt
        L_wall (pint.Quantity): thickness of the wall
        neutron_rate (pint.Quantity): neutron rate
        TBR (pint.Quantity): Tritium Breeding Ratio
        irradiations (list[tuple[pint.Quantity, pint.Quantity]]): list
            of tuples with the start and stop times of irradiations
        k_wall (pint.Quantity): mass transport coefficient for the
            walls
        k_top (pint.Quantity): mass transport coefficient for the
            top surface
        concentrations (list[pint.Quantity]): list of concentrations
            at each time step
        times (list[pint.Quantity]): list of times at each time step

    """

    def __init__(
        self,
        radius: pint.Quantity,
        height: pint.Quantity,
        TBR: pint.Quantity,
        neutron_rate: pint.Quantity,
        k_wall: pint.Quantity | Callable[[pint.Quantity], pint.Quantity],
        k_top: pint.Quantity | Callable[[pint.Quantity], pint.Quantity],
        irradiations: list,
    ) -> None:

        self.radius = radius
        self.height = height

        self.L_wall = 0.06 * ureg.inches

        self.neutron_rate = neutron_rate

        self.TBR = TBR
        self.irradiations = irradiations

        self.k_wall = k_wall
        self.k_top = k_top

        self.concentrations = []
        self.times = []

    @property
    def volume(self):
        """
        Calculate the volume of the tritium model.

        Returns:
            pint.Quantity: The volume calculated as the product of the top area (A_top) and the height.
        """
        return self.A_top * self.height

    @property
    def A_top(self):
        """
        Calculate the top surface area of a cylinder.
        This method calculates the area of the top surface of a cylinder using the formula:
        A = π * r^2

        .. note::
            This neglects the presence of a re-entrant heater (perfect cylinder).

        Returns:
            pint.Quantity: The top surface area of the cylinder.
        """

        return np.pi * self.radius**2

    @property
    def A_wall(self):
        """
        Calculate the surface area of the wall.
        This method computes the surface area of the wall based on the perimeter
        and height of the wall, and adds the bottom area.

        Returns:
            pint.Quantity: The total surface area of the wall.
        """

        perimeter_wall = 2 * np.pi * (self.radius + self.L_wall)
        return perimeter_wall * self.height + self.A_top

    def source(self, t):
        """
        Calculate the source term at a given time ``t``.
        This method iterates through the list of irradiations and checks if the
        given time ``t`` falls within any irradiation period. If it does, it returns
        the product of the Tritium Breeding Ratio (TBR) and the neutron rate.
        If ``t`` does not fall within any irradiation period, it returns zero.

        Args:
            t (pint.Quantity): The time at which to calculate the source term.

        Returns:
            pint.Quantity: The source term at time ``t``. This is the product of TBR and
            neutron rate if ``t`` is within an irradiation period, otherwise zero.
        """

        for irradiation in self.irradiations:
            irradiation_start = irradiation[0]
            irradiation_stop = irradiation[1]
            if irradiation_start < t < irradiation_stop:
                return self.TBR * self.neutron_rate
        return 0 * self.TBR * self.neutron_rate

    def Q_wall(self, t, c_salt):
        """
        Calculate the release rate of tritium through the wall.

        .. math::
            Q_\mathrm{wall} = A_\mathrm{wall} \ k_\mathrm{wall} \ c_\mathrm{salt}

        Args:
            c_salt (pint.Quantity): The concentration of tritium in the salt.

        Returns:
            pint.Quantity: The release rate of tritium through the wall.
        """
        if callable(self.k_wall):
            k_wall = self.k_wall(t)
        else:
            k_wall = self.k_wall
        return self.A_wall * k_wall * c_salt

    def Q_top(self, t, c_salt):
        """
        Calculate the release rate of tritium through the top surface of the salt.

        .. math::
            Q_\mathrm{top} = A_\mathrm{top} \ k_\mathrm{top} \ c_\mathrm{salt}

        Args:
            c_salt (pint.Quantity): The concentration of tritium in the salt.

        Returns:
            pint.Quantity: The release rate of tritium through the top.
        """
        if callable(self.k_top):
            k_top = self.k_top(t)
        else:
            k_top = self.k_top
        return self.A_top * k_top * c_salt

    def rhs(self, t, c):
        """
        Calculate the right-hand side of the ODE.

        Args:
            t (float): time
            c (float): salt concentration

        Returns:
            pint.Quantity: the rhs of the ODE
        """
        t *= ureg.s
        c *= ureg.particle * ureg.m**-3

        return self.volume.to(ureg.m**3) ** -1 * (
            self.source(t).to(ureg.particle * ureg.s**-1)
            - self.Q_wall(t, c).to(ureg.particle * ureg.s**-1)
            - self.Q_top(t, c).to(ureg.particle * ureg.s**-1)
        )

    def _generate_time_intervals(self, t_final):
        """
        Generate time intervals splitting the irradiations and non-irradiation periods.
        Example: if the irradiations are ``[(0, 10), (60, 70)]`` and ``t_final`` is 100,
        the time intervals will be ``[(0, 10), (10, 60), (60, 70), (70, 100)]``.

        Args:
            t_final (pint.Quantity): The final time of the simulation.

        Returns:
            list[tuple[pint.Quantity, pint.Quantity]]: A list of tuples with the start and stop times of each interval.
        """
        time_intervals = []
        previous_tf = None

        for irr in self.irradiations:
            t0 = irr[0]
            tf = min(irr[1], t_final)

            if previous_tf is not None:
                time_intervals.append((previous_tf, t0))
            time_intervals.append((t0, tf))
            previous_tf = tf

        # Add the final interval from the last tf to t_final
        if previous_tf is not None and previous_tf < t_final:
            time_intervals.append((previous_tf, t_final))

        return time_intervals

    def run(self, t_final):
        """
        Solves the ODE between 0 and ``t_final``.
        It first generates the different time intervals based on the irradiations and non-irradiation periods.
        Then, it solves the ODE for each interval with ``scipy.optimize.minimize`` and concatenates the results.
        The results are stored in the ``concentrations`` and ``times`` attributes.

        Args:
            t_final (pint.Quantity): The final time of the simulation.
        """
        concentration_units = ureg.particle * ureg.m**-3
        time_units = ureg.s
        initial_concentration = 0
        time_intervals = self._generate_time_intervals(t_final)

        for interval in time_intervals:
            t0 = interval[0].to(time_units).magnitude
            tf = interval[1].to(time_units).magnitude

            res = solve_ivp(
                fun=self.rhs,
                t_span=(t0, tf),
                y0=[initial_concentration],
                t_eval=np.linspace(t0, tf, 1000),
                # method="RK45",  # RK45 doesn't catch the end of irradiations properly... unless constraining the max_step
                # max_step=(0.5 * ureg.h).to(time_units).magnitude,
                # method="Radau",
                method="BDF",
            )
            self.times.append(res.t)
            self.concentrations.append(res.y[0])
            initial_concentration = res.y[0][-1]

        self.times = np.concatenate(self.times) * time_units
        self.concentrations = np.concatenate(self.concentrations) * concentration_units

    def reset(self):
        """
        Reset the model by resetting the ``concentrations`` and ``times``
        attributes to empty lists.
        """
        self.concentrations = []
        self.times = []

    def integrated_release_top(self):
        """
        Calculate the cumulative release of tritium through the top surface.

        Returns:
            ndarray: array with units same size as the number of time steps,
            the integrated release of tritium through the top surface.
        """
        top_release = self.Q_top(t=self.times, c_salt=self.concentrations)
        integrated_top = cumulative_trapezoid(
            top_release.to(ureg.particle * ureg.h**-1).magnitude,
            self.times.to(ureg.h).magnitude,
            initial=0,
        )
        integrated_top *= ureg.particle  # attach units
        return integrated_top

    def integrated_release_wall(self):
        """
        Calculate the cumulative release of tritium through the walls.

        Returns:
            ndarray: array with units same size as the number of time steps,
            the integrated release of tritium through the walls.
        """
        wall_release = self.Q_wall(t=self.times, c_salt=self.concentrations)
        integrated_wall = cumulative_trapezoid(
            wall_release.to(ureg.particle * ureg.h**-1).magnitude,
            self.times.to(ureg.h).magnitude,
            initial=0,
        )
        integrated_wall *= ureg.particle  # attach units
        return integrated_wall


def quantity_to_activity(Q):
    """Converts a quantity of tritium to activity.
    By multiplying the quantity by the specific activity and molar mass of tritium.

    Args:
        Q (pint.Quantity): the quantity of tritium

    Returns:
        pint.Quantity: the equivalent activity
    """
    return Q * SPECIFIC_ACT * MOLAR_MASS


def activity_to_quantity(A):
    """Converts an activity of tritium to quantity.
    By dividing the activity by the specific activity and molar mass of tritium.

    Args:
        A (pint.Quantity): the activity of tritium

    Returns:
        pint.Quantity: the equivalent quantity
    """
    return A / (SPECIFIC_ACT * MOLAR_MASS)
