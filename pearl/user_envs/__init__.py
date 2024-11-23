# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

try:
    from gymnasium.envs.registration import register
except ModuleNotFoundError:
    from gym.envs.registration import register

for game in ["Catcher", "FlappyBird", "Pixelcopter", "PuckWorld", "Pong"]:
    register(id="{}-PLE-v0".format(game), entry_point=f"gym_pygame.envs:{game}Env")


register(
    id="Pendulum-no-timeout-v1",
    entry_point="gymnasium.envs.classic_control.pendulum:PendulumEnv",
    max_episode_steps=int(1e9),
)

register(
    id="HalfCheetah-no-timeout-v4",
    entry_point="gymnasium.envs.mujoco.half_cheetah_v4:HalfCheetahEnv",
    max_episode_steps=int(1e9),
)

register(
    id="HalfCheetah-no-timeout-v5",
    entry_point="gymnasium.envs.mujoco.half_cheetah_v5:HalfCheetahEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Hopper-no-timeout-v4",
    entry_point="gymnasium.envs.mujoco.hopper_v4:HopperEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Hopper-no-timeout-v5",
    entry_point="gymnasium.envs.mujoco.hopper_v5:HopperEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Ant-no-timeout-v4",
    entry_point="gymnasium.envs.mujoco.ant_v4:AntEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Ant-no-timeout-v5",
    entry_point="gymnasium.envs.mujoco.ant_v5:AntEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Humanoid-no-timeout-v4",
    entry_point="gymnasium.envs.mujoco.humanoid_v4:HumanoidEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Humanoid-no-timeout-v5",
    entry_point="gymnasium.envs.mujoco.humanoid_v5:HumanoidEnv",
    max_episode_steps=int(1e9),
)

register(
    id="HumanoidStandup-no-timeout-v4",
    entry_point="gymnasium.envs.mujoco.humanoidstandup_v4:HumanoidStandupEnv",
    max_episode_steps=int(1e9),
)

register(
    id="HumanoidStandup-no-timeout-v5",
    entry_point="gymnasium.envs.mujoco.humanoidstandup_v5:HumanoidStandupEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Pusher-no-timeout-v4",
    entry_point="gymnasium.envs.mujoco.pusher_v4:PusherEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Pusher-no-timeout-v5",
    entry_point="gymnasium.envs.mujoco.pusher_v5:PusherEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Reacher-no-timeout-v4",
    entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Reacher-no-timeout-v5",
    entry_point="gymnasium.envs.mujoco.reacher_v5:ReacherEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Walker2d-no-timeout-v4",
    max_episode_steps=int(1e9),
    entry_point="gymnasium.envs.mujoco.walker2d_v4:Walker2dEnv",
)

register(
    id="Walker2d-no-timeout-v5",
    max_episode_steps=int(1e9),
    entry_point="gymnasium.envs.mujoco.walker2d_v5:Walker2dEnv",
)

register(
    id="Swimmer-no-timeout-v4",
    entry_point="gymnasium.envs.mujoco.swimmer_v4:SwimmerEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Swimmer-no-timeout-v5",
    entry_point="gymnasium.envs.mujoco.swimmer_v5:SwimmerEnv",
    max_episode_steps=int(1e9),
)

register(
    id="CartPole-no-timeout-v1",
    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Catcher-PLE-1000-v0",
    entry_point="gym_pygame.envs:CatcherEnv",
    max_episode_steps=1000,
)

register(
    id="Catcher-PLE-no-timeout-v0",
    entry_point="gym_pygame.envs:CatcherEnv",
    max_episode_steps=int(1e9),
)

register(
    id="FlappyBird-PLE-1000-v0",
    entry_point="gym_pygame.envs:FlappyBirdEnv",
    max_episode_steps=1000,
)

register(
    id="FlappyBird-PLE-no-timeout-v0",
    entry_point="gym_pygame.envs:FlappyBirdEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Pong-PLE-1000-v0",
    entry_point="gym_pygame.envs:PongEnv",
    max_episode_steps=1000,
)

register(
    id="Pong-PLE-no-timeout-v0",
    entry_point="gym_pygame.envs:PongEnv",
    max_episode_steps=int(1e9),
)

register(
    id="Pixelcopter-PLE-1000-v0",
    entry_point="gym_pygame.envs:PixelcopterEnv",
    max_episode_steps=1000,
)

register(
    id="Pixelcopter-PLE-no-timeout-v0",
    entry_point="gym_pygame.envs:PixelcopterEnv",
    max_episode_steps=int(1e9),
)

register(
    id="PuckWorld-PLE-1000-v0",
    entry_point="gym_pygame.envs:PuckWorldEnv",
    max_episode_steps=1000,
)

register(
    id="PuckWorld-PLE-no-timeout-v0",
    entry_point="gym_pygame.envs:PuckWorldEnv",
    max_episode_steps=int(1e9),
)
