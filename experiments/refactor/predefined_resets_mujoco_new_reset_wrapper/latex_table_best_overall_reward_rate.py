# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

# Sample input text
import sys

import numpy as np

assert len(sys.argv) == 2
data_file = sys.argv[1]

with open(data_file, "r") as file:
    input_text = file.read()

plot_regex = re.compile(
    r"draw plot (\d+), name: (\w+)_(\w+), type: learning_curve, num curves: 7\ndraw curve 0\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 1\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 2\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 3\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 4\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 5\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 6\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\n"
)

data_matches = plot_regex.findall(input_text)

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


data = {"ddpg": {}, "td3": {}, "csac": {}, "ppo": {}}
for match in data_matches:
    (
        _,
        task,
        algo,
        rwd_10_val,
        rwd_10_ste,
        rwd_100_val,
        rwd_100_ste,
        rwd_1000_val,
        rwd_1000_ste,
        eps_rwd_val,
        eps_rwd_ste,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = match
    data[algo][task] = (
        float(rwd_10_val),
        float(rwd_10_ste),
        float(rwd_100_val),
        float(rwd_100_ste),
        float(rwd_1000_val),
        float(rwd_1000_ste),
        float(eps_rwd_val),
        float(eps_rwd_ste),
    )


# Generate LaTeX code
def generate_latex_table(my_data):
    latex_code = r"\begin{table}[h]" + "\n"
    latex_code += r"\centering" + "\n"
    latex_code += r"\resizebox{\columnwidth}{!}{" + "\n"
    latex_code += r"\begin{tabular}{|c|c|c|c|}" + "\n"
    latex_code += r"\toprule" + "\n"
    latex_code += (
        " & ".join([f"{algorithm_to_name[algo]}" for algo in algorithms])
        + r" \\"
        + "\n"
    )

    latex_code += r"\midrule" + "\n"
    # latex_code += r"\multirow{5}{*}{\makecell{Reward rate}}" + "\n"
    for task in tasks:
        row = ""
        for algo in algorithms:
            if task in my_data[algo]:
                (
                    rwd_10_val,
                    rwd_10_ste,
                    rwd_100_val,
                    rwd_100_ste,
                    rwd_1000_val,
                    rwd_1000_ste,
                    eps_rwd_val,
                    eps_rwd_ste,
                ) = my_data[algo][task]
                v = max(max(rwd_10_val, rwd_100_val), rwd_1000_val)

                if v == rwd_10_val:
                    s = rwd_10_ste
                elif v == rwd_100_val:
                    s = rwd_100_ste
                else:
                    s = rwd_1000_ste
                tmp = [
                    (v, s),
                    (eps_rwd_val, eps_rwd_ste),
                ]
                sig = stats_significant(
                    v,
                    s,
                    eps_rwd_val,
                    eps_rwd_ste,
                )
                if (sig[0] or sig[1]) and eps_rwd_val < v:
                    row += f"\\textbf{{{v:.1f} $\pm$ {s:.1f}}}"
                elif (sig[0] or sig[1]) and eps_rwd_val >= v:
                    row += f"\\textcolor{{gray}}{{{v:.1f} $\pm$ {s:.1f}}}"
                else:
                    row += f"{v:.1f} $\pm$ {s:.1f}"
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


# Generate LaTeX code for both parts
print(generate_latex_table(data))
