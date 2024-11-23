# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from alphaex.sweeper import Sweeper

mpl.rcParams.update(mpl.rcParamsDefault)

# pyre-ignore-all-errors
DEBUG = False


def get_stats_learning_curves_to_print(data_x_runs, criterion):
    if "last10" in criterion:
        mean = data_x_runs[:, -10:].mean()
        ste = data_x_runs[:, -10:].mean(1).std(ddof=1) / np.sqrt(data_x_runs.shape[0])
    elif "last" in criterion:
        mean = data_x_runs[:, -1].mean()
        ste = data_x_runs[:, -1].std(ddof=1) / np.sqrt(data_x_runs.shape[0])
    elif "first" in criterion:
        mean = data_x_runs[:, 0].mean()
        ste = data_x_runs[:, 0].std(ddof=1) / np.sqrt(data_x_runs.shape[0])
    elif "second" in criterion:
        mean = data_x_runs[:, 1].mean()
        ste = data_x_runs[:, 1].std(ddof=1) / np.sqrt(data_x_runs.shape[0])
    elif "third" in criterion:
        mean = data_x_runs[:, 2].mean()
        ste = data_x_runs[:, 2].std(ddof=1) / np.sqrt(data_x_runs.shape[0])
    elif "all" in criterion:
        mean = data_x_runs.mean()
        ste = data_x_runs.mean(1).std(ddof=1) / np.sqrt(data_x_runs.shape[0])
    else:
        raise NotImplementedError
    return mean, ste


def all_lists_same_length(lst):
    if not lst:
        return True
    first_length = len(lst[0])
    return all(len(sublist) == first_length for sublist in lst)


class Plotter(object):
    def __init__(self, plot_config_file):
        self.plot_config_file = plot_config_file
        plt.rcParams["font.size"] = "16"
        mpl.rcParams["axes.spines.right"] = False
        mpl.rcParams["axes.spines.top"] = False
        self.default_number = 0.0

    def draw_learning_curve(self, plot_dict, curve_num):
        param_sweeper = Sweeper(plot_dict["exp_input_file"])
        num_runs = plot_dict["num_runs"]
        if DEBUG:
            print("parse plot dict ", plot_dict)
        param_setting_list = param_sweeper.search(plot_dict, num_runs)
        if DEBUG:
            print(f"get {len(param_setting_list)} param setting matches")
        assert len(param_setting_list) != 0, print(
            "input json file is ", param_sweeper.config_dict
        )
        (
            mean_curve_list,
            ste_curve_list,
            mean_to_print_list,
            ste_to_print_list,
            all_curves_list,
        ) = ([], [], [], [], [])
        nan_replacement = -1e8 if "max" in plot_dict["criterion"] else 1e8
        if "plot_percentage" in plot_dict:
            plot_percentage = plot_dict["plot_percentage"]
        else:
            plot_percentage = 1.0
        for param_setting in param_setting_list:
            if "suffix" in plot_dict:
                suffix = plot_dict["suffix"]
                file_names = [f"{id}_{suffix}.npy" for id in param_setting["ids"]]
            else:
                file_names = [f"{id}.npy" for id in param_setting["ids"]]
            data_x_runs = []
            for file_name in file_names:
                file_path = os.path.join(plot_dict["exp_output_dir"], file_name)
                try:
                    data_x_runs.append(
                        np.nan_to_num(np.load(file_path), nan=nan_replacement)
                    )
                except:
                    print(file_path + " not found")
            if not all_lists_same_length(data_x_runs):
                return
            if len(data_x_runs) == 0 or len(data_x_runs[0]) == 1:
                return

            data_x_runs = np.array(data_x_runs)
            if "reward_offset" in plot_dict:
                data_x_runs = data_x_runs - plot_dict["reward_offset"]
            data_x_runs = data_x_runs[:, : int(data_x_runs.shape[1] * plot_percentage)]
            mean_curve = data_x_runs.mean(0)
            ste_curve = data_x_runs.std(0, ddof=1) / np.sqrt(data_x_runs.shape[0])
            mean_to_print, ste_to_print = get_stats_learning_curves_to_print(
                data_x_runs, plot_dict["criterion"]
            )
            mean_curve_list.append(mean_curve)
            ste_curve_list.append(ste_curve)
            all_curves_list.append(data_x_runs)
            mean_to_print_list.append(mean_to_print)
            ste_to_print_list.append(ste_to_print)
        indices = np.argsort(mean_to_print_list)
        if "max" in plot_dict["criterion"]:
            chosen_idx = indices[-1]
        elif "min" in plot_dict["criterion"]:
            chosen_idx = indices[0]

        mean_curve_to_draw = mean_curve_list[chosen_idx]
        ste_curve_to_draw = ste_curve_list[chosen_idx]
        all_curves_to_draw = all_curves_list[chosen_idx]

        if DEBUG:
            print(
                f"draw curve correspond to param setting: ",
                param_setting_list[chosen_idx],
            )

        print(
            mean_to_print_list[chosen_idx],
            ste_to_print_list[chosen_idx],
        )
        if plot_dict.get("do_not_draw_curves", False):
            return
        x_list = np.arange(mean_curve_to_draw.shape[0])
        if plot_dict.get("draw_all_curves_only", False):
            for i in range(len(all_curves_to_draw)):
                plt.plot(
                    x_list,
                    all_curves_to_draw[i],
                    linewidth=1,
                    linestyle=plot_dict["linestyle"],
                )
            return
        if "curve_labels" in plot_dict:
            if "linestyle" in plot_dict and "curve_colors" in plot_dict:
                plt.plot(
                    x_list,
                    mean_curve_to_draw,
                    linewidth=1,
                    linestyle=plot_dict["linestyle"],
                    label=plot_dict["curve_labels"][curve_num],
                    color=plot_dict["curve_colors"][curve_num],
                )
            else:
                plt.plot(
                    x_list,
                    mean_curve_to_draw,
                    linewidth=1,
                    label=plot_dict["curve_labels"][curve_num],
                )
            plt.legend(loc=plot_dict["label_loc"])
        else:
            plt.plot(
                x_list,
                mean_curve_to_draw,
                linewidth=1,
                linestyle=plot_dict["linestyle"],
            )
        if "linestyle" in plot_dict and "curve_colors" in plot_dict:
            plt.fill_between(
                x_list,
                mean_curve_to_draw - ste_curve_to_draw,
                mean_curve_to_draw + ste_curve_to_draw,
                alpha=0.2,
                color=plot_dict["curve_colors"][curve_num],
            )
        else:
            plt.fill_between(
                x_list,
                mean_curve_to_draw - ste_curve_to_draw,
                mean_curve_to_draw + ste_curve_to_draw,
                alpha=0.2,
            )

    def plot(self):
        plot_sweeper = Sweeper(self.plot_config_file)
        self.plots_dir = plot_sweeper.config_dict["plots_dir"][0]
        for plot_num in range(len(plot_sweeper.config_dict["plots"])):
            plot = plot_sweeper.config_dict["plots"][plot_num]
            if "exp_input_file" in plot_sweeper.config_dict:
                plot["exp_input_file"] = plot_sweeper.config_dict["exp_input_file"]
            if "exp_output_dir" in plot_sweeper.config_dict:
                plot["exp_output_dir"] = plot_sweeper.config_dict["exp_output_dir"]
            if "baseline_exp_input_file" in plot_sweeper.config_dict:
                plot["baseline_exp_input_file"] = plot_sweeper.config_dict[
                    "baseline_exp_input_file"
                ]
            if "baseline_exp_output_dir" in plot_sweeper.config_dict:
                plot["baseline_exp_output_dir"] = plot_sweeper.config_dict[
                    "baseline_exp_output_dir"
                ]
            plot_name = plot["name"][0]
            plot_type = plot["type"][0]
            num_curves = plot["num_combinations"]
            print(
                f"draw plot {plot_num}, name: {plot_name}, type: {plot_type}, num curves: {num_curves}"
            )
            if "x-label" in plot:
                x_label = plot["x-label"][0]
                plt.xlabel(x_label)
            if "y-label" in plot:
                y_label = plot["y-label"][0]
                plt.ylabel(y_label, rotation="horizontal", ha="right", va="top")
            if "xmin" in plot and "xmax" in plot:
                plt.xlim(plot["xmin"][0], plot["xmax"][0])
            if "ymin" in plot and "ymax" in plot:
                plt.ylim(plot["ymin"][0], plot["ymax"][0])
            if "log 10 x axis" in plot and plot["log 10 x axis"][0] is True:
                plt.xscale("log")
            if "xticks" in plot:
                if "xticks_labels" in plot:
                    plt.xticks(np.array(plot["xticks"][0]), plot["xticks_labels"][0])
                else:
                    plt.xticks(plot["xticks"][0])

            for curve_num in range(num_curves):
                print(f"draw curve {curve_num}")
                plot_dict = dict()
                plot_sweeper.parse_helper(curve_num, plot, plot_dict)
                if plot_type == "learning_curve":
                    self.draw_learning_curve(plot_dict, curve_num)
                else:
                    raise NotImplementedError(
                        "plot.py only supports drawing learning curves."
                    )
            if plot_dict.get("do_not_draw_curves", False):
                continue
            plt.grid(axis="y", alpha=0.5, linestyle="--")
            if not os.path.exists(self.plots_dir):
                os.makedirs(self.plots_dir)
            plt.savefig(
                os.path.join(self.plots_dir, plot["name"][0] + ".pdf"),
                bbox_inches="tight",
            )
            plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument(
        "--plot-config-file",
        default="pearl/experiments/refactor/no_resets_mujoco/plot.json",
        type=str,
        help="specify the plot json file",
    )
    args = parser.parse_args()
    my_plotter = Plotter(args.plot_config_file)
    my_plotter.plot()
