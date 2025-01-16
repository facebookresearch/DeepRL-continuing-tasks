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
    r"draw plot (\d+), name: learning_curves_(\w+)_(\w+), type: learning_curve, num curves: 4\ndraw curve 0\n([\d.-]+) ([\d.-]+)\ndraw curve 1\n([\d.-]+) ([\d.-]+)\ndraw curve 2\n([\d.-]+) ([\d.-]+)\ndraw curve 3\n([\d.-]+) ([\d.-]+)"
)

matches = plot_regex.findall(input_text)
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
random_average_reward = {
    "ant": -5.3196252104,
    "halfcheetah": -5.18204632616,
    "hopper": -4.03147586188,
    "humanoid": -0.07463485668,
    "walker2d": -4.28692965053,
}
def stats_significant(val1, ste1, val2, ste2):
    t_stat = abs(val1 - val2) / np.sqrt(ste1 ** 2 + ste2 ** 2)
    degrees_of_freedom = 9 * (ste1 ** 2 + ste2 ** 2)**2 / (ste1 ** 4 + ste2 ** 4)  # assume 10 samples
    return t_stat > critical_t_values[int(degrees_of_freedom)] # p < 0.05, 1-sided t-test

data = {algo: {"TD": {}, "RVI": {}, "MA": {}} for algo in algorithms}

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
    26: 1.706,
    27: 1.703,
    28: 1.701,
    29: 1.699,
    30: 1.697,
    31: 1.6955,
    32: 1.6944,
    33: 1.6934,
    34: 1.6924,
    35: 1.6913,
    36: 1.6903,
    37: 1.6892,
    38: 1.6882,
    39: 1.6872,
    40: 1.6861,
    41: 1.6851,
    42: 1.6841,
    43: 1.6831,
    44: 1.6822,
    45: 1.6812,
    46: 1.6802,
    47: 1.6793,
    48: 1.6783,
    49: 1.6774,
    50: 1.6764
}

for match in matches:
    _, task, algo, val0, ste0, val1, ste1, val2, ste2, val3, ste3 = match
    if algo in algorithms and task in tasks:
        data[algo]["TD"][task] = ((float(val0) - random_average_reward[task]) / (
            float(val3) - random_average_reward[task]
        ) - 1, stats_significant(float(val0), float(ste0), float(val3), float(ste3)))
        data[algo]["RVI"][task] = ((float(val1) - random_average_reward[task]) / (
            float(val3) - random_average_reward[task]
        ) - 1, stats_significant(float(val1), float(ste1), float(val3), float(ste3)))
        data[algo]["MA"][task] = ((float(val2) - random_average_reward[task]) / (
            float(val3) - random_average_reward[task]
        ) - 1, stats_significant(float(val2), float(ste2), float(val3), float(ste3)))


# Generate LaTeX code
def generate_latex_table(data):
    algorithms = list(data.keys())
    latex_code = r"\begin{table}[h]" + "\n"
    latex_code += r"\centering" + "\n"
    latex_code += r"\resizebox{\columnwidth}{!}{" + "\n"
    latex_code += r"\begin{tabular}{l|ccc|ccc|ccc|ccc}" + "\n"
    latex_code += r"\toprule" + "\n"
    latex_code += (
        r"Task & "
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
        r"RC & "
        + " & ".join([f" TD & RVI & MA " for algo in algorithms])
        + r" \\"
        + "\n"
    )
    latex_code += r"\midrule" + "\n"
    for task in tasks:
        row = " & " +  task_to_name[task] + " & "
        for algo in algorithms:
            if algo in data and task in data[algo]["TD"]:
                rc_val, sig = data[algo]["TD"][task]
                rc_str = f"{rc_val * 100:.2f}"
                if sig == False:
                    row += f"\\textcolor{{gray}}{{{rc_str}}} &"
                else:
                    row += f"{rc_str} & "
            if algo in data and task in data[algo]["RVI"]:
                rc_val, sig = data[algo]["RVI"][task]
                rc_str = f"{rc_val * 100:.2f}"
                if sig == False:
                    row += f"\\textcolor{{gray}}{{{rc_str}}} &"
                else:
                    row += f"{rc_str} & "
            if algo in data and task in data[algo]["MA"]:
                rc_val, sig = data[algo]["MA"][task]
                rc_str = f"{rc_val * 100:.2f}"
                if sig == False:
                    row += f"\\textcolor{{gray}}{{{rc_str}}} &"
                else:
                    row += f"{rc_str} & "
        row = row[:-2] + r"\\" + "\n"
        latex_code += row

    latex_code += r"\bottomrule" + "\n"
    latex_code += r"\end{tabular}" + "\n"
    latex_code += r"}" + "\n"
    latex_code += (
        r"\caption{Sensitivity Curves for Various Tasks and Algorithms} " + "\n"
    )
    latex_code += r"\label{tab:sensitivity_curves_part} " + "\n"
    latex_code += r"\end{table}" + "\n"

    return latex_code


# Generate LaTeX code for both parts
latex_code_part = generate_latex_table(data)


print("Latex Code:")
print(latex_code_part)
