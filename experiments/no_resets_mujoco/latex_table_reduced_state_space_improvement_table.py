# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import numpy as np
import sys

# Sample input text
assert len(sys.argv) == 2
data_file = sys.argv[1]

with open(data_file, "r") as file:
    input_text = file.read()

# Regular expression to extract the necessary data
# plot_regex = re.compile(
#     r"draw plot (\d+), name: sensitivity_curve_(\w+)_(\w+), type: sensitivity_curve, num curves: 1\ndraw curve 0\nbaseline ([\d.-]+) ([\d.-]+)\nreward centering ([\d.-]+) ([\d.-]+)"
# )
plot_regex = re.compile(
    r"draw plot (\d+), name: learning_curves_(\w+)_(\w+), type: learning_curve, num curves: 2\ndraw curve 0\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 1\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\n"
)

matches = plot_regex.findall(input_text)
# Data organization
algorithms = ["ddpg", "td3", "csac", "ppo"]
tasks = [
    "swimmer",
    "humanoidstandup",
    "reacher",
    "pusher",
    "specialant",
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
}
random_average_reward = {
    "humanoidstandup": 35.70074146,
    "pusher": -1.59619093781,
    "reacher": -0.84029656837,
    "swimmer": 0.000643282893,
    "specialant": -0.5372274849,
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


data = {algo: {"improved percentage": {}} for algo in algorithms}
for match in matches:
    _, task, algo, ref_val, ref_ste, val, ste = match
    if algo in algorithms and task in tasks:
        data[algo]["improved percentage"][task] = (
            (float(val) - random_average_reward[task])
            / (float(ref_val) - random_average_reward[task])
            - 1,
            stats_significant(float(val), float(ste), float(ref_val), float(ref_ste)),
        )


# Generate LaTeX code
def generate_latex_table(data):
    algorithms = list(data.keys())
    latex_code = r"\begin{table}[h]" + "\n"
    latex_code += r"\centering" + "\n"
    latex_code += r"\begin{tabular}{lcccc}" + "\n"
    latex_code += r"\toprule" + "\n"
    latex_code += (
        r"Task & "
        + " & ".join([f"{algorithm_to_name[algo]}" for algo in algorithms])
        + r" \\"
        + "\n"
    )
    latex_code += r"\midrule" + "\n"
    for task in tasks:
        row = task_to_name[task] + " & "
        for algo in algorithms:
            if algo in data and task in data[algo]["improved percentage"]:
                rc_val, sig = data[algo]["improved percentage"][task]
                rc_str = f"{rc_val * 100:.2f}"
                if sig == False:
                    row += f"\\textcolor{{gray}}{{{rc_str}}} &"
                else:
                    row += f"{rc_str} & "
        row = row[:-2] + r"\\" + "\n"
        latex_code += row

    latex_code += r"\bottomrule" + "\n"
    latex_code += r"\end{tabular}" + "\n"
    latex_code += (
        r"\caption{Sensitivity Curves for Various Tasks and Algorithms} " + "\n"
    )
    latex_code += r"\label{tab:sensitivity_curves_part} " + "\n"
    latex_code += r"\end{table}" + "\n"

    return latex_code


# Generate LaTeX code for both parts
latex_code_part = generate_latex_table(
    {
        algo: {
            # "baseline": data[algo]["baseline"],
            "improved percentage": data[algo]["improved percentage"],
        }
        for algo in algorithms
    },
)


print(latex_code_part)
