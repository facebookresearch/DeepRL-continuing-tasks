# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

# Sample input text
import sys

import numpy as np

# assert len(sys.argv) == 2
# data_file = sys.argv[1]

# Data organization
algorithms = ["ddpg", "td3", "csac", "ppo"]
tasks = [
    "halfcheetah",
    "ant",
    "hopper",
    "humanoid",
    "walker2d",
]

algorithm_to_name = {
    "ddpg": "DDPG",
    "td3": "TD3",
    "csac": "SAC",
    "ppo": "PPO",
}
task_to_name = {
    "halfcheetah": "HalfCheetah",
    "ant": "Ant",
    "hopper": "Hopper",
    "humanoid": "Humanoid",
    "walker2d": "Walker2d",
}
critical_t_values = {
    1: 6.314,
    2: 2.920,
    3: 2.353,
    4: 2.132,
    5: 2.015,
    6: 1.943,
    7: 1.895,
    8: 1.860,
    9: 1.833,
    10: 1.812,
    11: 1.796,
    12: 1.782,
    13: 1.771,
    14: 1.761,
    15: 1.753,
    16: 1.746,
    17: 1.740,
    18: 1.734,
    19: 1.729,
    20: 1.725,
    21: 1.721,
    22: 1.717,
    23: 1.714,
    24: 1.711,
    25: 1.708,
}


def stats_significant(val1, ste1, val2, ste2, choose_max=True):
    if choose_max:
        v = max(val1, val2)
    else:
        v = min(val1, val2)
    if v == val1:
        s = ste1
    else:
        s = ste2

    tmp = [(val1, ste1), (val2, ste2)]
    rtv = []
    for i in range(len(tmp)):
        if s == 0 and tmp[i][1] == 0:
            sig = False if v == tmp[i][0] else True
            rtv.append(sig)
            continue
        t_stat = abs(v - tmp[i][0]) / np.sqrt(s**2 + tmp[i][1] ** 2)
        degrees_of_freedom = (
            9 * (s**2 + tmp[i][1] ** 2) ** 2 / (s**4 + tmp[i][1] ** 4)
        )  # assume 10 samples
        sig = (
            t_stat > critical_t_values[int(degrees_of_freedom)]
        )  # p < 0.05, 1-sided t-test
        rtv.append(sig)
    return rtv


def read_data(data_file, multiplier=1):
    with open(data_file, "r") as file:
        input_text = file.read()

    plot_regex = re.compile(
        r"draw plot (\d+), name: (\w+)_(\w+), type: learning_curve, num curves: 6\ndraw curve 0\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 1\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 2\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 3\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 4\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 5\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\n"
    )

    data_matches = plot_regex.findall(input_text)
    data = {"ddpg": {}, "td3": {}, "csac": {}, "ppo": {}}
    for match in data_matches:
        (
            _,
            task,
            algo,
            rwd_10_val,
            rwd_10_ste,
            predefined_rwd_10_val,
            predefined_rwd_10_ste,
            rwd_100_val,
            rwd_100_ste,
            predefined_rwd_100_val,
            predefined_rwd_100_ste,
            rwd_1000_val,
            rwd_1000_ste,
            predefined_rwd_1000_val,
            predefined_rwd_1000_ste,
        ) = match
        data[algo][task] = (
            float(rwd_10_val) * multiplier,
            float(rwd_10_ste) * multiplier,
            float(predefined_rwd_10_val) * multiplier,
            float(predefined_rwd_10_ste) * multiplier,
            float(rwd_100_val) * multiplier,
            float(rwd_100_ste) * multiplier,
            float(predefined_rwd_100_val) * multiplier,
            float(predefined_rwd_100_ste) * multiplier,
            float(rwd_1000_val) * multiplier,
            float(rwd_1000_ste) * multiplier,
            float(predefined_rwd_1000_val) * multiplier,
            float(predefined_rwd_1000_ste) * multiplier,
        )
    return data


# Generate LaTeX code
def generate_latex_table(my_data, choose_max=True):
    latex_code = r"\begin{table}[h]" + "\n"
    latex_code += r"\centering" + "\n"
    latex_code += r"\resizebox{\columnwidth}{!}{" + "\n"
    latex_code += r"\begin{tabular}{|c|l|cc|cc|cc|cc|}" + "\n"
    latex_code += r"\toprule" + "\n"
    latex_code += (
        r"& Algorithm & "
        + " & ".join(
            [
                f"\\multicolumn{{2}}{{c|}}{{{algorithm_to_name[algo]}}}"
                for algo in algorithms
            ]
        )
        + r" \\"
        + "\n"
    )
    latex_code += (
        r"& Reset & "
        + " & ".join([f"agent-controlled & predefined" for algo in algorithms])
        + r" \\"
        + "\n"
    )
    latex_code += r"\midrule" + "\n"
    latex_code += r"\multirow{5}{*}{\makecell{Reset cost = 10}}" + "\n"
    for task in tasks:
        row = " & " + task_to_name[task] + " & "
        for algo in algorithms:
            if task in my_data[algo]:
                (
                    rwd_10_val,
                    rwd_10_ste,
                    predefined_rwd_10_val,
                    predefined_rwd_10_ste,
                    rwd_100_val,
                    rwd_100_ste,
                    predefined_rwd_100_val,
                    predefined_rwd_100_ste,
                    rwd_1000_val,
                    rwd_1000_ste,
                    predefined_rwd_1000_val,
                    predefined_rwd_1000_ste,
                ) = my_data[algo][task]

                tmp = [
                    (rwd_10_val, rwd_10_ste),
                    (predefined_rwd_10_val, predefined_rwd_10_ste),
                ]
                sig = stats_significant(
                    rwd_10_val,
                    rwd_10_ste,
                    predefined_rwd_10_val,
                    predefined_rwd_10_ste,
                    choose_max=choose_max,
                )
                assert len(sig) == 2
                for i in range(len(sig)):
                    if sig[i]:
                        row += f"{tmp[i][0]:.1f} $\pm$ {tmp[i][1]:.1f}"
                    else:
                        row += f"\\textbf{{{tmp[i][0]:.1f} $\pm$ {tmp[i][1]:.1f}}}"
                    row += " & "
        row = row[:-2] + r"\\ " + "\n"
        latex_code += row
    latex_code = latex_code[:-2] + "\n"
    latex_code += r"\midrule" + "\n"
    latex_code += r"\multirow{5}{*}{\makecell{Reset cost = 100}}" + "\n"
    for task in tasks:
        row = " & " + task_to_name[task] + " & "
        for algo in algorithms:
            if task in my_data[algo]:
                (
                    rwd_10_val,
                    rwd_10_ste,
                    predefined_rwd_10_val,
                    predefined_rwd_10_ste,
                    rwd_100_val,
                    rwd_100_ste,
                    predefined_rwd_100_val,
                    predefined_rwd_100_ste,
                    rwd_1000_val,
                    rwd_1000_ste,
                    predefined_rwd_1000_val,
                    predefined_rwd_1000_ste,
                ) = my_data[algo][task]

                tmp = [
                    (rwd_100_val, rwd_100_ste),
                    (predefined_rwd_100_val, predefined_rwd_100_ste),
                ]
                sig = stats_significant(
                    rwd_100_val,
                    rwd_100_ste,
                    predefined_rwd_100_val,
                    predefined_rwd_100_ste,
                    choose_max=choose_max,
                )
                assert len(sig) == 2
                for i in range(len(sig)):
                    if sig[i]:
                        row += f"{tmp[i][0]:.1f} $\pm$ {tmp[i][1]:.1f}"
                    else:
                        row += f"\\textbf{{{tmp[i][0]:.1f} $\pm$ {tmp[i][1]:.1f}}}"
                    row += " & "
        row = row[:-2] + r"\\ " + "\n"
        latex_code += row
    latex_code = latex_code[:-2] + "\n"
    latex_code += r"\midrule" + "\n"
    latex_code += r"\multirow{5}{*}{\makecell{Reset cost = 1000}}" + "\n"
    for task in tasks:
        row = " & " + task_to_name[task] + " & "
        for algo in algorithms:
            if task in my_data[algo]:
                (
                    rwd_10_val,
                    rwd_10_ste,
                    predefined_rwd_10_val,
                    predefined_rwd_10_ste,
                    rwd_100_val,
                    rwd_100_ste,
                    predefined_rwd_100_val,
                    predefined_rwd_100_ste,
                    rwd_1000_val,
                    rwd_1000_ste,
                    predefined_rwd_1000_val,
                    predefined_rwd_1000_ste,
                ) = my_data[algo][task]

                tmp = [
                    (rwd_1000_val, rwd_1000_ste),
                    (predefined_rwd_1000_val, predefined_rwd_1000_ste),
                ]
                sig = stats_significant(
                    rwd_1000_val,
                    rwd_1000_ste,
                    predefined_rwd_1000_val,
                    predefined_rwd_1000_ste,
                    choose_max=choose_max,
                )
                assert len(sig) == 2
                for i in range(len(sig)):
                    if sig[i]:
                        row += f"{tmp[i][0]:.1f} $\pm$ {tmp[i][1]:.1f}"
                    else:
                        row += f"\\textbf{{{tmp[i][0]:.1f} $\pm$ {tmp[i][1]:.1f}}}"
                    row += " & "
        row = row[:-2] + r"\\ " + "\n"
        latex_code += row
    latex_code = latex_code[:-2] + "\n"

    latex_code += r"\bottomrule" + "\n"
    latex_code += r"\end{tabular}" + "\n"
    latex_code += r"}" + "\n"
    latex_code += (
        r"\caption{The table presents the reward rate and number of resets of the learned policies over 10,000 evaluation steps with varying reset costs. To ensure a fair comparison, the reset cost is excluded from the reward rate computation. The lower section of the table shows the number of resets during evaluation. The boldface represents the same meaning as in \Cref{tab: problem reset vs. episodic mujoco}. These results demonstrate that policies learned in tasks with higher reset costs generally lead to fewer resets. In several cases (e.g., DDPG in Humanoid), higher reset costs are also associated with higher reward rates.}"
        + "\n"
    )
    latex_code += r"\label{tab: different costs}" + "\n"
    latex_code += r"\end{table}" + "\n"

    return latex_code


# Generate LaTeX code
data = read_data(
    "/home/yiwan/pearl_execution/pearl/experiments/refactor/agent_resets_mujoco_new/plot_compare_predefined_resets_reward_rate.txt"
)
print(generate_latex_table(data))
# data = read_data(
#     "/home/yiwan/pearl_execution/pearl/experiments/refactor/agent_resets_mujoco_new/plot_compare_predefined_resets_reset_rate.txt",
#     multiplier=1e4,
# )
# print(generate_latex_table(data, choose_max=False))
# data = read_data(
#     "/home/yiwan/pearl_execution/pearl/experiments/refactor/agent_resets_mujoco_new/plot_compare_predefined_resets_reward_rate_excluding_resets.txt"
# )
# print(generate_latex_table(data))
