# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

import numpy as np
# Sample input text
import sys

assert len(sys.argv) == 3
discount_data_file = sys.argv[1]
offset_data_file = sys.argv[2]

with open(discount_data_file, "r") as file:
    discount_input_text = file.read()

with open(offset_data_file, "r") as file:
    offset_input_text = file.read()

plot_regex = re.compile(
    r"draw plot (\d+), name: learning_curves_(\w+)_(\w+), type: learning_curve, num curves: 5\ndraw curve 0\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 1\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 2\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 3\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 4\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\n"
)

discount_matches = plot_regex.findall(discount_input_text)

plot_regex = re.compile(
    r"draw plot (\d+), name: learning_curves_(\w+)_(\w+), type: learning_curve, num curves: 6\ndraw curve 0\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 1\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 2\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 3\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 4\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 5\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\n"
)

offset_matches = plot_regex.findall(offset_input_text)
# Data organization
algorithms = ["ddpg", "td3", "csac", "ppo"]
tasks = [
    "swimmer",
    "humanoidstandup",
    "reacher",
    "pusher",
    "specialant",
    "halfcheetah",
    "ant",
    "hopper",
    "humanoid",
    "walker2d",
    "halfcheetahar",
    "antar",
    "hopperar",
    "humanoidar",
    "walker2dar",
]
algorithm_to_name = {
    "ddpg": "DDPG",
    "td3": "TD3",
    "csac": "SAC",
    "ppo": "PPO",
}
task_to_name = {
    "swimmer": "Swimmer",
    "humanoidstandup": "HumanoidStandup",
    "reacher": "Reacher",
    "pusher": "Pusher",
    "specialant": "SpecialAnt",
    "halfcheetah": "HalfCheetah",
    "ant": "Ant",
    "hopper": "Hopper",
    "humanoid": "Humanoid",
    "walker2d": "Walker2d",
    "halfcheetahar": "HalfCheetah",
    "antar": "Ant",
    "hopperar": "Hopper",
    "humanoidar": "Humanoid",
    "walker2dar": "Walker2d",
}
random_average_reward = {
    "ant": -0.44703722719,
    "halfcheetah": -0.30164630014,
    "hopper": 0.37602373998,
    "humanoid": 4.60982662389,
    "walker2d": -0.3429192951,
    "humanoidstandup": 35.70074146,
    "pusher": -1.59619093781,
    "reacher": -0.84029656837,
    "swimmer": 0.000643282893,
    "specialant": -0.5372274849,
    "antar": -5.3196252104,
    "halfcheetahar": -5.18204632616,
    "hopperar": -4.03147586188,
    "humanoidar": -0.07463485668,
    "walker2dar": -4.28692965053,
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

def stats_significant(val1, ste1, val2, ste2):
    t_stat = abs(val1 - val2) / np.sqrt(ste1 ** 2 + ste2 ** 2)
    degrees_of_freedom = 9 * (ste1 ** 2 + ste2 ** 2)**2 / (ste1 ** 4 + ste2 ** 4)  # assume 10 samples
    return t_stat > critical_t_values[int(degrees_of_freedom)] # p < 0.05, 1-sided t-test

discount_data = {
    algo: {"0.999": {}} for algo in algorithms
}
for match in discount_matches:
    _, task, algo, val0, ste0, val1, ste1, val2, ste2, val3, ste3, val4, ste4 = match
    if algo in algorithms and task in tasks:
        if float(val3) - random_average_reward[task] < 0:
            discount_data[algo]["0.999"][task] = (None, None)
        else:
            discount_data[algo]["0.999"][task] = (
                (float(val4) - random_average_reward[task])
                / (float(val3) - random_average_reward[task])
                - 1,
                stats_significant(float(val4), float(ste4), float(val3), float(ste3)),
            )

offset_data = {
    algo: {"100": {}} for algo in algorithms
}
for match in offset_matches:
    (
        _,
        task,
        algo,
        val0,
        ste0,
        val1,
        ste1,
        val2,
        ste2,
        val3,
        ste3,
        val4,
        ste4,
        val5,
        ste5,
    ) = match
    if algo in algorithms and task in tasks:
        if float(val5) - random_average_reward[task] < 0:
            offset_data[algo]["100"][task] = (None, None)
        else:
            offset_data[algo]["100"][task] = (
                (float(val4) - random_average_reward[task])
                / (float(val5) - random_average_reward[task])
                - 1,
                stats_significant(float(val4), float(ste4), float(val5), float(ste5)),
            )

# Generate LaTeX code
def generate_latex_table(discount_data, offset_data):
    latex_code = r"\begin{table}[h]" + "\n"
    latex_code += r"\centering" + "\n"
    latex_code += r"\resizebox{\columnwidth}{!}{" + "\n"
    latex_code += r"\begin{tabular}{|c|l|c|c|c|c|c|c|c|c|c|}" + "\n"
    latex_code += r"\toprule" + "\n"
    latex_code += (
        r" & & \multicolumn{4}{c|}{Discount factor $0.99 \to 0.999$} & \multicolumn{4}{c|}{All rewards $+ 100$} \\"
        + "\n"
    )
    latex_code += r"\midrule" + "\n"
    latex_code += (
        r"& Algorithm & DDPG & TD3 & SAC & PPO & DDPG & TD3 & SAC & PPO \\"
        + "\n"
    )
    latex_code += r"\midrule" + "\n"
    for task in tasks:
        if task == "swimmer":
            latex_code += r"\multirow{5}{*}{\makecell{No \\ resets}}" + "\n"
        elif task == "halfcheetah":
            latex_code += r"\midrule" + "\n"
            latex_code += r"\multirow{5}{*}{\makecell{Predefined \\ resets}}" + "\n"
        elif task == "halfcheetahar":
            latex_code += r"\midrule" + "\n"
            latex_code += r"\multirow{5}{*}{\makecell{Agent-controlled \\ resets}}" + "\n"
        row = " & " + task_to_name[task] + " & "
        for algo in algorithms:
            if algo in discount_data and task in discount_data[algo]["0.999"]:
                rc_val, sig = discount_data[algo]["0.999"][task]
                rc_str = f"{rc_val * 100:.2f}" if rc_val is not None else "N/A"
                if sig == False:
                    row += f"\\textcolor{{gray}}{{{rc_str}}} &"
                else:
                    row += f"{rc_str} & "
        for algo in algorithms:
            if algo in offset_data and task in offset_data[algo]["100"]:
                rc_val, sig = offset_data[algo]["100"][task]
                rc_str = f"{rc_val * 100:.2f}" if rc_val is not None else "N/A"
                if sig == False:
                    row += f"\\textcolor{{gray}}{{{rc_str}}} &"
                else:
                    row += f"{rc_str} & "
        row = row[:-2] + r"\\ " + "\n"
        latex_code += row
    latex_code = latex_code[:-2] + "\n"

    latex_code += r"\bottomrule" + "\n"
    latex_code += r"\end{tabular}" + "\n"
    latex_code += r"}" + "\n"
    latex_code += (
        r"\caption{A large discount factor or reward offset hurt all tested algorithms' performance.}"
        + "\n"
    )
    latex_code += r"\label{tab: combined table}" + "\n"
    latex_code += r"\end{table}" + "\n"

    return latex_code


# Generate LaTeX code for both parts
print(generate_latex_table(discount_data, offset_data))
