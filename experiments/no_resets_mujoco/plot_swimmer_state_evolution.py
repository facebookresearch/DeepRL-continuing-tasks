# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams["font.size"] = "12"
reward_rate_no_resets, reward_rate_resets = [], []
for run in range(10):
    reward_rate_no_resets.append(np.load(
        os.getcwd() + f"/experiments/no_resets_mujoco/{168*run + 3}_average_reward.npy"
    ))
    reward_rate_resets.append(np.load(
        os.getcwd() + f"/experiments/no_resets_mujoco/{168*run + 59}_eval_average_reward.npy"
    ))
print("no resets")
for run in range(10):
    print(f"run {run}, reward rate last 100000 steps ", reward_rate_no_resets[run][-10:].mean())
print("reset w.p. 0.001")
for run in range(10):
    print(f"run {run}, reward rate last 100000 steps ", reward_rate_resets[run][-10:].mean())
print("plot no resets, discount factor 0.999, run 1")
reward_rate_no_resets_sample_run = np.load(
    os.getcwd() + f"/experiments/no_resets_mujoco/171_visited_observations.npy"
)
print("plot reset w.p. 0.001, discount factor 0.999, run 0")
reward_rate_resets_sample_run = np.load(
    os.getcwd() + f"/experiments/no_resets_mujoco/59_visited_observations.npy"
)

for i in range(4):
    concat = np.concatenate((
        reward_rate_no_resets_sample_run[250 * i:250 * (i+1)], 
        reward_rate_resets_sample_run[250 * i:250 * (i+1)]
        ))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    color_map = ["#1f77b4", "#ff7f0e"]
    color = np.concatenate((np.zeros(250), np.ones(250)))
    y = np.concatenate((np.zeros(250), np.ones(250)))
    ax.scatter(
        xs=concat[:250, 0],
        ys=concat[:250, 1],
        zs=concat[:250, 2],
        color=[color_map[0]],
        label="no resets",
    )
    ax.scatter(
        xs=concat[250:, 0],
        ys=concat[250:, 1],
        zs=concat[250:, 2],
        color=[color_map[1]],
        label="random resets",
    )
    ax.set_xlim(-70, 10)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel("angle of the front tip")
    ax.set_ylabel("angle of the first rotor")
    ax.set_zlabel("angle of the second rotor")
    # if i == 0:
    #     ax.legend(loc='upper right')
    os.makedirs(os.getcwd() + f"/experiments/no_resets_mujoco/swimmer_state_evolution", exist_ok=True)
    plt.savefig(os.getcwd() + f"/experiments/no_resets_mujoco/swimmer_state_evolution/swimmer_{i}.pdf", bbox_inches='tight', pad_inches=0.4)
    plt.close()
