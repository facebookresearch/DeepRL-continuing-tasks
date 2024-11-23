# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import re
import numpy as np
import sys

# Sample input text
assert len(sys.argv) == 2
data_file = sys.argv[1]

with open(data_file, "r") as file:
    input_text = file.read()

# Regular expression to extract the necessary data
plot_regex = re.compile(
    r"draw plot (\d+), name: learning_curves_(\w+)_(\w+), type: learning_curve, num curves: 4\ndraw curve 0\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 1\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 2\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 3\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\n"
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
        if float(val3) - random_average_reward[task] < 0:
            data[algo]["TD"][task] = (None, None)
            data[algo]["RVI"][task] = (None, None)
            data[algo]["MA"][task] = (None, None)
        else:
            data[algo]["TD"][task] = ((float(val0) - random_average_reward[task]) / (
                float(val3) - random_average_reward[task]
            ) - 1, stats_significant(float(val0), float(ste0), float(val3), float(ste3)))
            data[algo]["RVI"][task] = ((float(val1) - random_average_reward[task]) / (
                float(val3) - random_average_reward[task]
            ) - 1, stats_significant(float(val1), float(ste1), float(val3), float(ste3)))
            data[algo]["MA"][task] = ((float(val2) - random_average_reward[task]) / (
                float(val3) - random_average_reward[task]
            ) - 1, stats_significant(float(val2), float(ste2), float(val3), float(ste3)))

for algo in data:
    for rc in data[algo]:
        total_improvement = 0
        total_tasks = 0
        for task in data[algo][rc]:
            if data[algo][rc][task][0] is not None:
                total_improvement += data[algo][rc][task][0]
                total_tasks += 1
        data[algo][rc]["average_improvement"] = total_improvement / total_tasks


# Generate LaTeX code
def generate_latex_table(data, only_TD=False):
    algorithms = list(data.keys())
    latex_code = r"\begin{table}[h]" + "\n"
    latex_code += r"\centering" + "\n"
    latex_code += r"\resizebox{\columnwidth}{!}{" + "\n"
    if only_TD:
        latex_code += r"\begin{tabular}{|c|l|c|c|c|c|}" + "\n"
    else:
        latex_code += r"\begin{tabular}{|c|l|ccc|ccc|ccc|ccc|}" + "\n"
    latex_code += r"\toprule" + "\n"
    if not only_TD:
        # latex_code += r"& Algorithm & & DDPG & & & TD3 & & & SAC & & & PPO & \\" + "\n"
        latex_code += (
        r"& Algorithm & "
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
            r"& RC approach & "
            + " & ".join([f" TD & RVI & MA " for algo in algorithms])
            + r" \\"
            + "\n"
        )
    else:
        latex_code += r"& Algorithm & DDPG & TD3 & SAC & PPO \\" + "\n"
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
        row = " & " +  task_to_name[task] + " & "
        for algo in algorithms:
            if algo in data and task in data[algo]["TD"]:
                rc_val, sig = data[algo]["TD"][task]
                rc_str = f"{rc_val * 100:.2f}" if rc_val is not None else "N/A"
                if sig == False:
                    row += f"\\textcolor{{gray}}{{{rc_str}}} &"
                else:
                    row += f"{rc_str} & "
            if not only_TD:
                if algo in data and task in data[algo]["RVI"]:
                    rc_val, sig = data[algo]["RVI"][task]
                    rc_str = f"{rc_val * 100:.2f}" if rc_val is not None else "N/A"
                    if sig == False:
                        row += f"\\textcolor{{gray}}{{{rc_str}}} &"
                    else:
                        row += f"{rc_str} & "
                if algo in data and task in data[algo]["MA"]:
                    rc_val, sig = data[algo]["MA"][task]
                    rc_str = f"{rc_val * 100:.2f}" if rc_val is not None else "N/A"
                    if sig == False:
                        row += f"\\textcolor{{gray}}{{{rc_str}}} &"
                    else:
                        row += f"{rc_str} & "
        row = row[:-2] + r"\\" + "\n"
        latex_code += row
    latex_code += r"\midrule" + "\n"
    row  = r"& Average improvement"
    for algo in algorithms:
        row += " & " + f"{data[algo]['TD']['average_improvement'] * 100:.2f}"
        if not only_TD:
            row += " & " + f"{data[algo]['RVI']['average_improvement'] * 100:.2f}"
            row += " & " + f"{data[algo]['MA']['average_improvement'] * 100:.2f}"
    latex_code += row + r" \\" + "\n"

    latex_code += r"\bottomrule" + "\n"
    latex_code += r"\end{tabular}" + "\n"
    latex_code += r"}" + "\n"
    if only_TD:
        latex_code += ("\caption{Percentage of reward rate improvement when applying reward centering to the tested algorithms in Mujoco testbeds} " + "\n")
        latex_code += r"\label{tab: reward centering mujoco} " + "\n"
    else:
        latex_code += (
            r"\caption{Performance improvement when applying reward centering to the tested algorithms to solve Mujoco testbeds. Here, RVI standards for the reference-state-based approach. MA standards for the moving-average-based approach.} " + "\n"
        )
        latex_code += r"\label{tab: reward centering mujoco all} " + "\n"
    latex_code += r"\end{table}" + "\n"

    return latex_code


# Generate LaTeX code for both parts

print("% All")
print(generate_latex_table(data))

print("% TD-based reward centering")
print(generate_latex_table(data, only_TD=True))
