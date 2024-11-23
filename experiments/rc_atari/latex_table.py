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
plot_regex = re.compile(
    r"draw plot (\d+), name: learning_curves_(\w+)_(\w+), type: learning_curve, num curves: 4\ndraw curve 0\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 1\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 2\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 3\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\n"
)
matches = plot_regex.findall(input_text)

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
algo_to_name = {
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
random_average_reward = {
    "breakout": -0.018306,
    "pong": -0.02187725,
    "spaceinvader": 0.013011,
    "beamrider": 0.00414625,
    "seaquest": 0.00036025,
    "mspacman": 0.05666175,
}

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
}
# task_map = {task: i for i, task in enumerate(tasks)}
# print(task_map)
# data = {algo: {"baseline": [], "reward_centering": []} for algo in algorithms}
def stats_significant(val1, ste1, val2, ste2):
    t_stat = abs(val1 - val2) / np.sqrt(ste1 ** 2 + ste2 ** 2)
    degrees_of_freedom = 9 * (ste1 ** 2 + ste2 ** 2)**2 / (ste1 ** 4 + ste2 ** 4)  # assume 10 samples
    # print(degrees_of_freedom)
    # print(t_stat, t_stat > 1.734, val1, val2, ste1, ste2)
    return t_stat > critical_t_values[int(degrees_of_freedom)] # p < 0.05, 1-sided t-test
    
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

for algo in data:
    for rc in data[algo]:
        total_improvement = 0
        total_tasks = 0
        for task in data[algo][rc]:
            total_improvement += data[algo][rc][task][0]
            total_tasks += 1
        data[algo][rc]["average_improvement"] = total_improvement / total_tasks

# Generate LaTeX code
def generate_latex_table(data, only_TD=False):
    latex_code = r"\begin{table}[h]" + "\n"
    latex_code += r"\centering" + "\n"
    latex_code += r"\resizebox{\columnwidth}{!}{" + "\n"
    if only_TD:
        latex_code += r"\begin{tabular}{|l|c|c|c|}" + "\n"
    else:
        latex_code += r"\begin{tabular}{|l|ccc|ccc|ccc|}" + "\n"
    latex_code += r"\toprule" + "\n"
    if not only_TD:
        latex_code += (
            r"Algorithm & & "
            + " & & & ".join([f"{algo_to_name[algo]}" for algo in algorithms])
            + r" & \\"
            + "\n"
        )
        latex_code += (
            r"RC approach & "
            + " & ".join([f" TD & RVI & MA " for algo in algorithms])
            + r" \\"
            + "\n"
        )
    else:
        latex_code += (
            r"Algorithm & "
            + " & ".join([f"{algo_to_name[algo]}" for algo in algorithms])
            + r" \\"
            + "\n"
        )
    latex_code += r"\midrule" + "\n"
    for task in tasks:
        row = task_to_name[task] + " & "
        for algo in algorithms:
            if algo in data and task in data[algo]["TD"]:
                rc_val, sig = data[algo]["TD"][task]
                rc_str = f"{rc_val * 100:.2f}"
                if sig == False:
                    row += f"\\textcolor{{gray}}{{{rc_str}}} &"
                else:
                    row += f"{rc_str} & "
            if not only_TD:
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

    latex_code += r"\midrule" + "\n"
    row  = r"Average improvement"
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
        latex_code += r"\caption{Percentage of reward rate improvement when applying reward centering to the tested algorithms in Atari tasks. Statistically significant improvement percentage numbers are marked in boldface.} " + "\n"
        latex_code += r"\label{tab: reward centering atari} " + "\n"
    else:
        latex_code += (
            r"\caption{The performance improvement when applying reward centering in the tested algorithms to solve the Atari testbeds.} " + "\n"
        )
        latex_code += r"\label{tab: reward centering atari all} " + "\n"
    latex_code += r"\end{table}" + "\n"

    return latex_code


# Generate LaTeX code for both parts
print("% All")
print(generate_latex_table(data))

print("% TD-based reward centering")
print(generate_latex_table(data, only_TD=True))
