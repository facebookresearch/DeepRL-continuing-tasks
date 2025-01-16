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
data_file = "/home/yiwan/pearl_execution/pearl/experiments/refactor/predefined_resets_atari_new_reset_wrapper/plot_compare_reset_costs_reward_rate.txt"

with open(data_file, "r") as file:
    input_text = file.read()

plot_regex = re.compile(
    r"draw plot (\d+), name: (\w+)_(\w+), type: learning_curve, num curves: 9\ndraw curve 0\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 1\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 2\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 3\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 4\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 5\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 6\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 7\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 8\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\n"
)

data_matches = plot_regex.findall(input_text)

# Data organization
algorithms = ["dqn", "sac", "ppo"]
tasks = [
    "breakout",
    "pong",
    "spaceinvader",
    "beamrider",
    "seaquest",
    "mspacman",
]
algorithm_to_name = {
    "dqn": "DQN",
    "sac": "SAC",
    "ppo": "PPO",
}
task_to_name = {
    "breakout": "Breakout",
    "pong": "Pong",
    "spaceinvader": "SpaceInvader",
    "beamrider": "BeamRider",
    "seaquest": "Seaquest",
    "mspacman": "MsPacman",
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


def stats_significant(val1, ste1, val2, ste2, val3, ste3, choose_max=True):
    if choose_max:
        v = max(max(val1, val2), val3)
    else:
        v = min(min(val1, val2), val3)
    if v == val1:
        s = ste1
    elif v == val2:
        s = ste2
    else:
        s = ste3
    tmp = [(val1, ste1), (val2, ste2), (val3, ste3)]
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


data = {"dqn": {}, "sac": {}, "ppo": {}}
for match in data_matches:
    (
        _,
        task,
        algo,
        rwd_10_val_eval_10,
        rwd_10_ste_eval_10,
        rwd_100_val_eval_10,
        rwd_100_ste_eval_10,
        rwd_1000_val_eval_10,
        rwd_1000_ste_eval_10,
        rwd_10_val_eval_100,
        rwd_10_ste_eval_100,
        rwd_100_val_eval_100,
        rwd_100_ste_eval_100,
        rwd_1000_val_eval_100,
        rwd_1000_ste_eval_100,
        rwd_10_val_eval_1000,
        rwd_10_ste_eval_1000,
        rwd_100_val_eval_1000,
        rwd_100_ste_eval_1000,
        rwd_1000_val_eval_1000,
        rwd_1000_ste_eval_1000,
    ) = match
    data[algo][task] = (
        float(rwd_10_val_eval_10) * 100,
        float(rwd_10_ste_eval_10) * 100,
        float(rwd_100_val_eval_10) * 100,
        float(rwd_100_ste_eval_10) * 100,
        float(rwd_1000_val_eval_10) * 100,
        float(rwd_1000_ste_eval_10) * 100,
        float(rwd_10_val_eval_100) * 100,
        float(rwd_10_ste_eval_100) * 100,
        float(rwd_100_val_eval_100) * 100,
        float(rwd_100_ste_eval_100) * 100,
        float(rwd_1000_val_eval_100) * 100,
        float(rwd_1000_ste_eval_100) * 100,
        float(rwd_10_val_eval_1000) * 100,
        float(rwd_10_ste_eval_1000) * 100,
        float(rwd_100_val_eval_1000) * 100,
        float(rwd_100_ste_eval_1000) * 100,
        float(rwd_1000_val_eval_1000) * 100,
        float(rwd_1000_ste_eval_1000) * 100,
    )


# Generate LaTeX code
def generate_latex_table(my_data):
    latex_code = r"\begin{table}[h]" + "\n"
    latex_code += r"\centering" + "\n"
    latex_code += r"\resizebox{\columnwidth}{!}{" + "\n"
    latex_code += r"\begin{tabular}{|c|l|ccc|ccc|ccc|ccc|}" + "\n"
    latex_code += r"\toprule" + "\n"
    latex_code += (
        r" & Algorithm & "
        + " & ".join(
            [
                f"\\multicolumn{{3}}{{c|}}{{{algorithm_to_name[algo]}}}"
                for algo in algorithms
            ]
        )
        + r" \\"
        + "\n"
    )
    latex_code += (
        r" & Solution reset cost & "
        + " & ".join([f"10 & 100 & 1000" for algo in algorithms])
        + r" \\"
        + "\n"
    )
    latex_code += r"\midrule" + "\n"
    latex_code += r"\multirow{5}{*}{\makecell{Problem reset cost = 10}}" + "\n"
    for task in tasks:
        row = " & " + task_to_name[task] + " & "
        for algo in algorithms:
            if task in my_data[algo]:
                (
                    rwd_10_val_eval_10,
                    rwd_10_ste_eval_10,
                    rwd_100_val_eval_10,
                    rwd_100_ste_eval_10,
                    rwd_1000_val_eval_10,
                    rwd_1000_ste_eval_10,
                    rwd_10_val_eval_100,
                    rwd_10_ste_eval_100,
                    rwd_100_val_eval_100,
                    rwd_100_ste_eval_100,
                    rwd_1000_val_eval_100,
                    rwd_1000_ste_eval_100,
                    rwd_10_val_eval_1000,
                    rwd_10_ste_eval_1000,
                    rwd_100_val_eval_1000,
                    rwd_100_ste_eval_1000,
                    rwd_1000_val_eval_1000,
                    rwd_1000_ste_eval_1000,
                ) = my_data[algo][task]
                tmp = [
                    (rwd_10_val_eval_10, rwd_10_ste_eval_10),
                    (rwd_100_val_eval_10, rwd_100_ste_eval_10),
                    (rwd_1000_val_eval_10, rwd_1000_ste_eval_10),
                ]
                sig = stats_significant(
                    rwd_10_val_eval_10,
                    rwd_10_ste_eval_10,
                    rwd_100_val_eval_10,
                    rwd_100_ste_eval_10,
                    rwd_1000_val_eval_10,
                    rwd_1000_ste_eval_10,
                )  # binary vector of length 3, each element indicates whether the corresponding value is whether stat significant different from the highest value
                assert len(sig) == 3
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
    latex_code += r"\multirow{5}{*}{\makecell{Problem reset cost = 100}}" + "\n"
    for task in tasks:
        row = " & " + task_to_name[task] + " & "
        for algo in algorithms:
            if task in my_data[algo]:
                (
                    rwd_10_val_eval_10,
                    rwd_10_ste_eval_10,
                    rwd_100_val_eval_10,
                    rwd_100_ste_eval_10,
                    rwd_1000_val_eval_10,
                    rwd_1000_ste_eval_10,
                    rwd_10_val_eval_100,
                    rwd_10_ste_eval_100,
                    rwd_100_val_eval_100,
                    rwd_100_ste_eval_100,
                    rwd_1000_val_eval_100,
                    rwd_1000_ste_eval_100,
                    rwd_10_val_eval_1000,
                    rwd_10_ste_eval_1000,
                    rwd_100_val_eval_1000,
                    rwd_100_ste_eval_1000,
                    rwd_1000_val_eval_1000,
                    rwd_1000_ste_eval_1000,
                ) = my_data[algo][task]
                tmp = [
                    (rwd_10_val_eval_100, rwd_10_ste_eval_100),
                    (rwd_100_val_eval_100, rwd_100_ste_eval_100),
                    (rwd_1000_val_eval_100, rwd_1000_ste_eval_100),
                ]
                sig = stats_significant(
                    rwd_10_val_eval_100,
                    rwd_10_ste_eval_100,
                    rwd_100_val_eval_100,
                    rwd_100_ste_eval_100,
                    rwd_1000_val_eval_100,
                    rwd_1000_ste_eval_100,
                )  # binary vector of length 3, each element indicates whether the corresponding value is whether stat significant different from the highest value
                assert len(sig) == 3
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
    latex_code += r"\multirow{5}{*}{\makecell{Problem reset cost = 1000}}" + "\n"
    for task in tasks:
        row = " & " + task_to_name[task] + " & "
        for algo in algorithms:
            if task in my_data[algo]:
                (
                    rwd_10_val_eval_10,
                    rwd_10_ste_eval_10,
                    rwd_100_val_eval_10,
                    rwd_100_ste_eval_10,
                    rwd_1000_val_eval_10,
                    rwd_1000_ste_eval_10,
                    rwd_10_val_eval_100,
                    rwd_10_ste_eval_100,
                    rwd_100_val_eval_100,
                    rwd_100_ste_eval_100,
                    rwd_1000_val_eval_100,
                    rwd_1000_ste_eval_100,
                    rwd_10_val_eval_1000,
                    rwd_10_ste_eval_1000,
                    rwd_100_val_eval_1000,
                    rwd_100_ste_eval_1000,
                    rwd_1000_val_eval_1000,
                    rwd_1000_ste_eval_1000,
                ) = my_data[algo][task]
                tmp = [
                    (rwd_10_val_eval_1000, rwd_10_ste_eval_1000),
                    (rwd_100_val_eval_1000, rwd_100_ste_eval_1000),
                    (rwd_1000_val_eval_1000, rwd_1000_ste_eval_1000),
                ]
                sig = stats_significant(
                    rwd_10_val_eval_1000,
                    rwd_10_ste_eval_1000,
                    rwd_100_val_eval_1000,
                    rwd_100_ste_eval_1000,
                    rwd_1000_val_eval_1000,
                    rwd_1000_ste_eval_1000,
                )  # binary vector of length 3, each element indicates whether the corresponding value is whether stat significant different from the highest value
                assert len(sig) == 3
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


# # Generate LaTeX code
# def generate_latex_table2(my_data):
#     latex_code = r"\begin{table}[h]" + "\n"
#     latex_code += r"\centering" + "\n"
#     latex_code += r"\resizebox{\columnwidth}{!}{" + "\n"
#     latex_code += r"\begin{tabular}{|c|l|ccc|ccc|ccc|ccc|}" + "\n"
#     latex_code += r"\toprule" + "\n"
#     latex_code += (
#         r" & Algorithm & "
#         + " & ".join(
#             [
#                 f"\\multicolumn{{3}}{{c|}}{{{algorithm_to_name[algo]}}}"
#                 for algo in algorithms
#             ]
#         )
#         + r" \\"
#         + "\n"
#     )
#     latex_code += r"\midrule" + "\n"
#     latex_code += r"\multirow{5}{*}{\makecell{Problem reset cost = 10}}" + "\n"
#     for task in tasks:
#         row = " & " + task_to_name[task] + " & "
#         for algo in algorithms:
#             if task in my_data[algo]:
#                 (
#                     rwd_10_val_eval_10,
#                     rwd_10_ste_eval_10,
#                     rwd_100_val_eval_10,
#                     rwd_100_ste_eval_10,
#                     rwd_1000_val_eval_10,
#                     rwd_1000_ste_eval_10,
#                     rwd_10_val_eval_100,
#                     rwd_10_ste_eval_100,
#                     rwd_100_val_eval_100,
#                     rwd_100_ste_eval_100,
#                     rwd_1000_val_eval_100,
#                     rwd_1000_ste_eval_100,
#                     rwd_10_val_eval_1000,
#                     rwd_10_ste_eval_1000,
#                     rwd_100_val_eval_1000,
#                     rwd_100_ste_eval_1000,
#                     rwd_1000_val_eval_1000,
#                     rwd_1000_ste_eval_1000,
#                 ) = my_data[algo][task]
#                 v = max(
#                     max(rwd_10_val_eval_10, rwd_100_val_eval_10), rwd_1000_val_eval_10
#                 )

#                 if v == rwd_10_val_eval_10:
#                     s = rwd_10_ste_eval_10
#                 elif v == rwd_100_val_eval_10:
#                     s = rwd_100_ste_eval_10
#                 else:
#                     s = rwd_1000_ste_eval_10
#                 tmp = [
#                     (v, s),
#                     (eps_rwd_val, eps_rwd_ste),
#                 ]
#                 sig = stats_significant(
#                     v,
#                     s,
#                     eps_rwd_val,
#                     eps_rwd_ste,
#                 )
#                 if (sig[0] or sig[1]) and eps_rwd_val < v:
#                     row += f"\\textbf{{{v:.1f} $\pm$ {s:.1f}}}"
#                 elif (sig[0] or sig[1]) and eps_rwd_val >= v:
#                     row += f"\\textcolor{{gray}}{{{v:.1f} $\pm$ {s:.1f}}}"
#                 else:
#                     row += f"{v:.1f} $\pm$ {s:.1f}"
#                 row += " & "
#         row = row[:-2] + r"\\ " + "\n"
#         latex_code += row
#     latex_code = latex_code[:-2] + "\n"

#     latex_code += r"\midrule" + "\n"
#     latex_code += r"\multirow{5}{*}{\makecell{Problem reset cost = 100}}" + "\n"
#     for task in tasks:
#         row = " & " + task_to_name[task] + " & "
#         for algo in algorithms:
#             if task in my_data[algo]:
#                 (
#                     rwd_10_val_eval_10,
#                     rwd_10_ste_eval_10,
#                     rwd_100_val_eval_10,
#                     rwd_100_ste_eval_10,
#                     rwd_1000_val_eval_10,
#                     rwd_1000_ste_eval_10,
#                     rwd_10_val_eval_100,
#                     rwd_10_ste_eval_100,
#                     rwd_100_val_eval_100,
#                     rwd_100_ste_eval_100,
#                     rwd_1000_val_eval_100,
#                     rwd_1000_ste_eval_100,
#                     rwd_10_val_eval_1000,
#                     rwd_10_ste_eval_1000,
#                     rwd_100_val_eval_1000,
#                     rwd_100_ste_eval_1000,
#                     rwd_1000_val_eval_1000,
#                     rwd_1000_ste_eval_1000,
#                 ) = my_data[algo][task]
#                 tmp = [
#                     (rwd_10_val_eval_100, rwd_10_ste_eval_100),
#                     (rwd_100_val_eval_100, rwd_100_ste_eval_100),
#                     (rwd_1000_val_eval_100, rwd_1000_ste_eval_100),
#                 ]
#                 sig = stats_significant(
#                     rwd_10_val_eval_100,
#                     rwd_10_ste_eval_100,
#                     rwd_100_val_eval_100,
#                     rwd_100_ste_eval_100,
#                     rwd_1000_val_eval_100,
#                     rwd_1000_ste_eval_100,
#                 )  # binary vector of length 3, each element indicates whether the corresponding value is whether stat significant different from the highest value
#                 assert len(sig) == 3
#                 for i in range(len(sig)):
#                     if sig[i]:
#                         row += f"{tmp[i][0]:.1f} $\pm$ {tmp[i][1]:.1f}"
#                     else:
#                         row += f"\\textbf{{{tmp[i][0]:.1f} $\pm$ {tmp[i][1]:.1f}}}"
#                     row += " & "
#         row = row[:-2] + r"\\ " + "\n"
#         latex_code += row
#     latex_code = latex_code[:-2] + "\n"

#     latex_code += r"\midrule" + "\n"
#     latex_code += r"\multirow{5}{*}{\makecell{Problem reset cost = 1000}}" + "\n"
#     for task in tasks:
#         row = " & " + task_to_name[task] + " & "
#         for algo in algorithms:
#             if task in my_data[algo]:
#                 (
#                     rwd_10_val_eval_10,
#                     rwd_10_ste_eval_10,
#                     rwd_100_val_eval_10,
#                     rwd_100_ste_eval_10,
#                     rwd_1000_val_eval_10,
#                     rwd_1000_ste_eval_10,
#                     rwd_10_val_eval_100,
#                     rwd_10_ste_eval_100,
#                     rwd_100_val_eval_100,
#                     rwd_100_ste_eval_100,
#                     rwd_1000_val_eval_100,
#                     rwd_1000_ste_eval_100,
#                     rwd_10_val_eval_1000,
#                     rwd_10_ste_eval_1000,
#                     rwd_100_val_eval_1000,
#                     rwd_100_ste_eval_1000,
#                     rwd_1000_val_eval_1000,
#                     rwd_1000_ste_eval_1000,
#                 ) = my_data[algo][task]
#                 tmp = [
#                     (rwd_10_val_eval_1000, rwd_10_ste_eval_1000),
#                     (rwd_100_val_eval_1000, rwd_100_ste_eval_1000),
#                     (rwd_1000_val_eval_1000, rwd_1000_ste_eval_1000),
#                 ]
#                 sig = stats_significant(
#                     rwd_10_val_eval_1000,
#                     rwd_10_ste_eval_1000,
#                     rwd_100_val_eval_1000,
#                     rwd_100_ste_eval_1000,
#                     rwd_1000_val_eval_1000,
#                     rwd_1000_ste_eval_1000,
#                 )  # binary vector of length 3, each element indicates whether the corresponding value is whether stat significant different from the highest value
#                 assert len(sig) == 3
#                 for i in range(len(sig)):
#                     if sig[i]:
#                         row += f"{tmp[i][0]:.1f} $\pm$ {tmp[i][1]:.1f}"
#                     else:
#                         row += f"\\textbf{{{tmp[i][0]:.1f} $\pm$ {tmp[i][1]:.1f}}}"
#                     row += " & "
#         row = row[:-2] + r"\\ " + "\n"
#         latex_code += row
#     latex_code = latex_code[:-2] + "\n"

#     latex_code += r"\bottomrule" + "\n"
#     latex_code += r"\end{tabular}" + "\n"
#     latex_code += r"}" + "\n"
#     latex_code += (
#         r"\caption{The table presents the reward rate and number of resets of the learned policies over 10,000 evaluation steps with varying reset costs. To ensure a fair comparison, the reset cost is excluded from the reward rate computation. The lower section of the table shows the number of resets during evaluation. The boldface represents the same meaning as in \Cref{tab: problem reset vs. episodic mujoco}. These results demonstrate that policies learned in tasks with higher reset costs generally lead to fewer resets. In several cases (e.g., DDPG in Humanoid), higher reset costs are also associated with higher reward rates.}"
#         + "\n"
#     )
#     latex_code += r"\label{tab: different costs}" + "\n"
#     latex_code += r"\end{table}" + "\n"

#     return latex_code


# Generate LaTeX code for both parts
print(generate_latex_table(data))
