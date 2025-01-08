import matplotlib.pyplot as plt
import numpy as np

from libra_toolbox.tritium import ureg
from libra_toolbox.tritium.model import Model, quantity_to_activity
from libra_toolbox.tritium.lsc_measurements import LIBRASample, GasStream

import pint
from typing import List

COLLECTION_VOLUME = 10 * ureg.ml
LSC_SAMPLE_VOLUME = 10 * ureg.ml


def plot_bars(
    measurements: List[LIBRASample] | GasStream | dict,
    index=None,
    bar_width=0.35,
    stacked=True,
):
    if isinstance(measurements, dict):
        return plot_bars_old(measurements, index, bar_width, stacked)

    if isinstance(measurements, GasStream):
        measurements = measurements.samples

    vial_1_vals = ureg.Quantity.from_list(
        [sample.samples[0].activity for sample in measurements]
    )
    vial_2_vals = ureg.Quantity.from_list(
        [sample.samples[1].activity for sample in measurements]
    )
    vial_3_vals = ureg.Quantity.from_list(
        [sample.samples[2].activity for sample in measurements]
    )
    vial_4_vals = ureg.Quantity.from_list(
        [sample.samples[3].activity for sample in measurements]
    )

    if index is None:
        if stacked:
            index = np.arange(len(measurements))
        else:
            group_spacing = 1  # Adjust this value to control spacing between groups
            index = (
                np.arange(len(measurements)) * (group_spacing / 2 + 1) * bar_width * 4
            )

    if stacked:
        vial_3_bar = plt.bar(
            index,
            vial_3_vals,
            bar_width,
            label="Vial 3",
            color="#FB8500",
        )
        vial_4_bar = plt.bar(
            index,
            vial_4_vals,
            bar_width,
            label="Vial 4",
            color="#FFB703",
            bottom=vial_3_vals,
        )
        vial_1_bar = plt.bar(
            index,
            vial_1_vals,
            bar_width,
            label="Vial 1",
            color="#219EBC",
            bottom=vial_3_vals + vial_4_vals,
        )
        vial_2_bar = plt.bar(
            index,
            vial_2_vals,
            bar_width,
            label="Vial 2",
            color="#8ECAE6",
            bottom=vial_3_vals + vial_4_vals + vial_1_vals,
        )
    else:
        if isinstance(index, pint.Quantity) and not isinstance(
            bar_width, pint.Quantity
        ):
            raise TypeError(
                f"index and bar_width must be of the same type, got {index=}, {bar_width=}"
            )

        vial_1_bar = plt.bar(
            index - 1.5 * bar_width,
            vial_1_vals,
            bar_width,
            linewidth=2,
            edgecolor="white",
            label="Vial 1",
            color="#219EBC",
        )
        vial_2_bar = plt.bar(
            index - 0.5 * bar_width,
            vial_2_vals,
            bar_width,
            linewidth=2,
            edgecolor="white",
            label="Vial 2",
            color="#8ECAE6",
        )
        vial_3_bar = plt.bar(
            index + 0.5 * bar_width,
            vial_3_vals,
            bar_width,
            linewidth=2,
            edgecolor="white",
            label="Vial 3",
            color="#FB8500",
        )
        vial_4_bar = plt.bar(
            index + 1.5 * bar_width,
            vial_4_vals,
            bar_width,
            linewidth=2,
            edgecolor="white",
            label="Vial 4",
            color="#FFB703",
        )

    return index


def plot_bars_old(measurements, index=None, bar_width=0.35, stacked=True):
    vial_1_vals = (
        np.array([sample[1].magnitude for sample in measurements.values()]) * ureg.Bq
    )
    vial_2_vals = (
        np.array([sample[2].magnitude for sample in measurements.values()]) * ureg.Bq
    )
    vial_3_vals = (
        np.array([sample[3].magnitude for sample in measurements.values()]) * ureg.Bq
    )
    vial_4_vals = (
        np.array([sample[4].magnitude for sample in measurements.values()]) * ureg.Bq
    )

    if index is None:
        if stacked:
            index = np.arange(len(measurements))
        else:
            group_spacing = 1  # Adjust this value to control spacing between groups
            index = (
                np.arange(len(measurements)) * (group_spacing / 2 + 1) * bar_width * 4
            )

    if stacked:
        vial_3_bar = plt.bar(
            index,
            vial_3_vals,
            bar_width,
            label="Vial 3",
            color="#FB8500",
        )
        vial_4_bar = plt.bar(
            index,
            vial_4_vals,
            bar_width,
            label="Vial 4",
            color="#FFB703",
            bottom=vial_3_vals,
        )
        vial_1_bar = plt.bar(
            index,
            vial_1_vals,
            bar_width,
            label="Vial 1",
            color="#219EBC",
            bottom=vial_3_vals + vial_4_vals,
        )
        vial_2_bar = plt.bar(
            index,
            vial_2_vals,
            bar_width,
            label="Vial 2",
            color="#8ECAE6",
            bottom=vial_3_vals + vial_4_vals + vial_1_vals,
        )
    else:
        vial_1_bar = plt.bar(
            index - 1.5 * bar_width,
            vial_1_vals,
            bar_width,
            linewidth=2,
            edgecolor="white",
            label="Vial 1",
            color="#219EBC",
        )
        vial_2_bar = plt.bar(
            index - 0.5 * bar_width,
            vial_2_vals,
            bar_width,
            linewidth=2,
            edgecolor="white",
            label="Vial 2",
            color="#8ECAE6",
        )
        vial_3_bar = plt.bar(
            index + 0.5 * bar_width,
            vial_3_vals,
            bar_width,
            linewidth=2,
            edgecolor="white",
            label="Vial 3",
            color="#FB8500",
        )
        vial_4_bar = plt.bar(
            index + 1.5 * bar_width,
            vial_4_vals,
            bar_width,
            linewidth=2,
            edgecolor="white",
            label="Vial 4",
            color="#FFB703",
        )

    return index


def replace_water(sample_activity, time, replacement_times):
    sample_activity_changed = np.copy(sample_activity)
    times_changed = np.copy(time)

    for replacement_time in sorted(replacement_times):
        indices = np.where(times_changed > replacement_time)
        # at each replacement, make the sample activity drop to zero
        sample_activity_changed[indices] -= sample_activity_changed[indices][0]

        # insert nan value to induce a line break in plots
        if indices[0].size > 0:
            first_index = indices[0][0]
            sample_activity_changed = np.insert(
                sample_activity_changed, first_index, np.nan * ureg.Bq
            )
            times_changed = np.insert(times_changed, first_index, np.nan * ureg.day)

    return sample_activity_changed, times_changed


def plot_sample_activity_top(
    model: Model,
    replacement_times,
    collection_vol=COLLECTION_VOLUME,
    lsc_sample_vol=LSC_SAMPLE_VOLUME,
    **kwargs,
):
    integrated_top = quantity_to_activity(model.integrated_release_top()).to(ureg.Bq)
    sample_activity_top = integrated_top / collection_vol * lsc_sample_vol
    times = model.times
    sample_activity_top, times = replace_water(
        sample_activity_top, times, replacement_times
    )
    l = plt.plot(times.to(ureg.day), sample_activity_top, **kwargs)
    return l


def plot_sample_activity_wall(
    model: Model,
    replacement_times,
    collection_vol=COLLECTION_VOLUME,
    lsc_sample_vol=LSC_SAMPLE_VOLUME,
    **kwargs,
):
    integrated_wall = quantity_to_activity(model.integrated_release_wall()).to(ureg.Bq)
    sample_activity_wall = integrated_wall / collection_vol * lsc_sample_vol
    times = model.times
    sample_activity_wall, times = replace_water(
        sample_activity_wall, times, replacement_times
    )
    l = plt.plot(times.to(ureg.day), sample_activity_wall, **kwargs)
    return l


def plot_salt_inventory(model: Model, **kwargs):
    salt_inventory = (quantity_to_activity(model.concentrations) * model.volume).to(
        ureg.Bq
    )
    l = plt.plot(model.times.to(ureg.day), salt_inventory, **kwargs)
    return l


def plot_top_release(model: Model, **kwargs):
    top_release = model.Q_top(t=model.times, c_salt=model.concentrations)
    top_release = quantity_to_activity(top_release).to(ureg.Bq * ureg.h**-1)
    l = plt.plot(model.times.to(ureg.day), top_release, **kwargs)
    return l


def plot_wall_release(model: Model, **kwargs):
    wall_release = model.Q_wall(t=model.times, c_salt=model.concentrations)
    wall_release = quantity_to_activity(wall_release).to(ureg.Bq * ureg.h**-1)
    l = plt.plot(model.times.to(ureg.day), wall_release, **kwargs)
    return l


def plot_integrated_top_release(model: Model, **kwargs):
    integrated_top = quantity_to_activity(model.integrated_release_top()).to(ureg.Bq)
    sample_activity_top = integrated_top / COLLECTION_VOLUME * LSC_SAMPLE_VOLUME
    l = plt.plot(model.times.to(ureg.day), sample_activity_top, **kwargs)
    return l


def plot_integrated_wall_release(model: Model, **kwargs):
    integrated_wall = quantity_to_activity(model.integrated_release_wall()).to(ureg.Bq)
    sample_activity_wall = integrated_wall / COLLECTION_VOLUME * LSC_SAMPLE_VOLUME
    l = plt.plot(model.times.to(ureg.day), sample_activity_wall, **kwargs)
    return l


def plot_irradiation(model: Model, **kwargs):
    pols = []
    for irr in model.irradiations:
        pol = plt.axvspan(irr[0].to(ureg.day), irr[1].to(ureg.day), **kwargs)
        pols.append(pol)
    return pols
