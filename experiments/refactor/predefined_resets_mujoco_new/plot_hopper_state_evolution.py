# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = "12"
# reward_rate_no_resets, reward_rate_resets = [], []
# for run in range(10):
#     reward_rate_no_resets.append(
#         np.load(
#             os.path.expanduser(
#                 f"~/mydata/out/refactor/predefined_resets_mujoco_new/{168*run + 3}_average_reward.npy"
#             )
#         )
#     )
#     reward_rate_resets.append(
#         np.load(
#             os.path.expanduser(
#                 f"~/mydata/out/refactor/predefined_resets_mujoco_new/{168*run + 59}_eval_average_reward.npy"
#             )
#         )
#     )
# print("no resets")
# for run in range(10):
#     print(
#         f"run {run}, reward rate last 100000 steps ",
#         reward_rate_no_resets[run][-10:].mean(),
#     )
# print("reset w.p. 0.001")
# for run in range(10):
#     print(
#         f"run {run}, reward rate last 100000 steps ",
#         reward_rate_resets[run][-10:].mean(),
#     )
for run in range(0, 10):
    print("plot 10, run 0")
    reward_rate_no_resets_sample_run = np.load(
        os.path.expanduser(
            f"~/mydata/out/refactor/predefined_resets_mujoco_new/{38 + 280 * run}_visited_observations.npy"
        )
    )
    print("plot 100, run 0")
    reward_rate_resets_sample_run = np.load(
        os.path.expanduser(
            f"~/mydata/out/refactor/predefined_resets_mujoco_new/{78 + 280 * run}_visited_observations.npy"
        )
    )
    reward_rate_resets_sample_run2 = np.load(
        os.path.expanduser(
            f"~/mydata/out/refactor/predefined_resets_mujoco_new/{118 + 280 * run}_visited_observations.npy"
        )
    )
    print(np.var(reward_rate_no_resets_sample_run, axis=0)[:5].mean())
    print(np.var(reward_rate_resets_sample_run, axis=0)[:5].mean())
    print(np.var(reward_rate_resets_sample_run2, axis=0)[:5].mean())

for i in range(4):
    concat = np.concatenate(
        (
            reward_rate_no_resets_sample_run[250 * i : 250 * (i + 1)],
            reward_rate_resets_sample_run[250 * i : 250 * (i + 1)],
        )
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    color_map = ["#1f77b4", "#ff7f0e"]
    color = np.concatenate((np.zeros(250), np.ones(250)))
    y = np.concatenate((np.zeros(250), np.ones(250)))
    ax.scatter(
        xs=concat[:250, 3],
        ys=concat[:250, 1],
        zs=concat[:250, 2],
        color=[color_map[0]],
        label="no resets",
    )
    ax.scatter(
        xs=concat[250:, 3],
        ys=concat[250:, 1],
        zs=concat[250:, 2],
        color=[color_map[1]],
        label="random resets",
    )
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("angle leg")
    ax.set_ylabel("angle torse")
    ax.set_zlabel("angle thigh")
    # if i == 0:
    #     ax.legend(loc='upper right')
    os.makedirs(
        os.path.expanduser(
            f"~/pearl_execution/pearl/experiments/refactor/predefined_resets_mujoco_new/hopper_state_evolution"
        ),
        exist_ok=True,
    )
    plt.savefig(
        os.path.expanduser(
            f"~/pearl_execution/pearl/experiments/refactor/predefined_resets_mujoco_new/hopper_state_evolution/{i}.pdf"
        ),
        bbox_inches="tight",
        pad_inches=0.4,
    )
    plt.close()
