# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

# Sample input text
import sys

import numpy as np

assert len(sys.argv) == 3
reward_rate_data_file = sys.argv[1]
num_resets_data_file = sys.argv[2]

with open(reward_rate_data_file, "r") as file:
    reward_rate_input_text = file.read()

with open(num_resets_data_file, "r") as file:
    num_resets_input_text = file.read()

plot_regex = re.compile(
    r"draw plot (\d+), name: learning_curves_(\w+)_(\w+), type: learning_curve, num curves: 2\ndraw curve 0\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\ndraw curve 1\n([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+) ([\d.-]+|-?[1-9]\d*\.?\d*e[-+]?\d+)\n"
)

reward_rate_matches = plot_regex.findall(reward_rate_input_text)
num_resets_matches = plot_regex.findall(num_resets_input_text)
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


def stats_significant(val1, ste1, val2, ste2):
    t_stat = abs(val1 - val2) / np.sqrt(ste1**2 + ste2**2)
    degrees_of_freedom = (
        9 * (ste1**2 + ste2**2) ** 2 / (ste1**4 + ste2**4)
    )  # assume 10 samples
    return (
        t_stat > critical_t_values[int(degrees_of_freedom)]
    )  # p < 0.05, 1-sided t-test


reward_rate_data = {"ddpg": {}, "td3": {}, "csac": {}, "ppo": {}}
for match in reward_rate_matches:
    (
        _,
        task,
        algo,
        agent_controlled_val,
        agent_controlled_ste,
        predefined_val,
        predefined_ste,
    ) = match
    reward_rate_data[algo][task] = (
        float(agent_controlled_val),
        float(agent_controlled_ste),
        float(predefined_val),
        float(predefined_ste),
    )

num_resets_data = {"ddpg": {}, "td3": {}, "csac": {}, "ppo": {}}
for match in num_resets_matches:
    (
        _,
        task,
        algo,
        agent_controlled_val,
        agent_controlled_ste,
        predefined_val,
        predefined_ste,
    ) = match
    num_resets_data[algo][task] = (
        float(agent_controlled_val),
        float(agent_controlled_ste),
        float(predefined_val),
        float(predefined_ste),
    )


# Generate LaTeX code
def generate_latex_table(reward_rate_data, num_resets_data):
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
        r"& & "
        + " & ".join([f"agent-controlled & predefined" for algo in algorithms])
        + r" \\"
        + "\n"
    )
    latex_code += r"\midrule" + "\n"
    latex_code += r"\multirow{5}{*}{\makecell{Reward \\ rate}}" + "\n"
    for task in tasks:
        row = " & " + task_to_name[task] + " & "
        for algo in algorithms:
            if task in reward_rate_data[algo]:
                continuing_val, continuing_ste, episodic_val, episodic_ste = (
                    reward_rate_data[algo][task]
                )
                sig = stats_significant(
                    continuing_val, continuing_ste, episodic_val, episodic_ste
                )
                if not sig:
                    row += (
                        f"\\textbf{{{continuing_val:.2f} $\pm$ {continuing_ste:.2f}}} & \\textbf{{{episodic_val:.2f} $\pm$ {episodic_ste:.2f}}}"
                        + " & "
                    )
                elif continuing_val > episodic_val:
                    row += (
                        f"\\textbf{{{continuing_val:.2f} $\pm$ {continuing_ste:.2f}}}"
                        + " & "
                        + f"{episodic_val:.2f} $\pm$ {episodic_ste:.2f} & "
                    )
                else:
                    row += (
                        f"{continuing_val:.2f} $\pm$ {continuing_ste:.2f} & "
                        + f"\\textbf{{{episodic_val:.2f} $\pm$ {episodic_ste:.2f}}}"
                        + " & "
                    )
        row = row[:-2] + r"\\ " + "\n"
        latex_code += row
    latex_code = latex_code[:-2] + "\n"

    latex_code += r"\midrule" + "\n"
    latex_code += r"\multirow{5}{*}{\makecell{Number \\ of resets}}" + "\n"
    for task in tasks:
        row = " & " + task_to_name[task] + " & "
        for algo in algorithms:
            if task in num_resets_data[algo]:
                continuing_val, continuing_ste, episodic_val, episodic_ste = (
                    num_resets_data[algo][task]
                )
                sig = stats_significant(
                    continuing_val, continuing_ste, episodic_val, episodic_ste
                )
                if not sig:
                    row += (
                        f"\\textbf{{{continuing_val:.2f} $\pm$ {continuing_ste:.2f}}} & \\textbf{{{episodic_val:.2f} $\pm$ {episodic_ste:.2f}}}"
                        + " & "
                    )
                elif continuing_val < episodic_val:
                    row += (
                        f"\\textbf{{{continuing_val:.2f} $\pm$ {continuing_ste:.2f}}}"
                        + " & "
                        + f"{episodic_val:.2f} $\pm$ {episodic_ste:.2f} & "
                    )
                else:
                    row += (
                        f"{continuing_val:.2f} $\pm$ {continuing_ste:.2f} & "
                        + f"\\textbf{{{episodic_val:.2f} $\pm$ {episodic_ste:.2f}}}"
                        + " & "
                    )
        row = row[:-2] + r"\\ " + "\n"
        latex_code += row
    latex_code = latex_code[:-2] + "\n"

    latex_code += r"\bottomrule" + "\n"
    latex_code += r"\end{tabular}" + "\n"
    latex_code += r"}" + "\n"
    latex_code += (
        r"\caption{A comparison of policies learned in testbeds with predefined resets versus those learned in testbeds with agent-controlled resets. The upper group shows the mean and the standard error of the reward rates when deploying learned policies obtained in these two settings for $10,000$ steps. The higher reward rate is highlighted in bold, and if the difference is statistically insignificant, both values are also marked in bold. The lower group shows the number of resets within the evaluation steps, with the fewer number of resets indicated in bold. In general, algorithms achieve a higher reward rate and lower reset frequency when running on testbeds with predefined resets compared to those where resets are controlled by the agent.}"
        + "\n"
    )
    latex_code += r"\label{tab: problem reset vs. episodic mujoco}" + "\n"
    latex_code += r"\end{table}" + "\n"

    return latex_code


# Generate LaTeX code for both parts
print(generate_latex_table(reward_rate_data, num_resets_data))
