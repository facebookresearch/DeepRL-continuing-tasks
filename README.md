# Deep Reinforcement Learning in Continuing Tasks

This repository provides the codebase for the paper *An Empirical Study of Deep Reinforcement Learning in Continuing Tasks*. The paper explores challenges that continuing tasks present to current deep reinforcement learning (RL) algorithms using a suite of continuing task testbeds. It empirically demonstrates the effectiveness of several reward-centering techniques that improve the performance of all studied algorithms on these continuing testbeds.

The code is based on the existing RL package [Pearl](https://github.com/facebookresearch/Pearl/), which itself is built with [PyTorch](https://pytorch.org/). The testbeds are based on Mujoco and Atari environments provided by [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/). Experiments use [AlphaEx](https://github.com/AmiiThinks/AlphaEx) for configuration sweeping.

## Why studying deep RL in continuing tasks?

Continuing tasks refer to tasks where the agent-environment interaction is ongoing and can not be broken down into episodes. These tasks are suitable when environment resets are unavailable, agent-controlled, or predefined but where all rewards—including those beyond resets—are critical. These scenarios frequently occur in real-world applications and can not be modeled by episodic tasks.
While modern deep RL algorithms have been extensively studied and well understood in episodic tasks, their behavior in continuing tasks remains underexplored.

## Why studying the reward centering technique in these testbeds?

Recent research ([source](https://arxiv.org/pdf/2405.09999)) shows that discounted RL methods for solving continuing
tasks can perform significantly better if they center their rewards by subtracting
out the rewards’ empirical average. Empirical analysis of the paper primarily focused on a temporal-difference-based reward centering method in conjunction with Q-learning. Our paper extends their findings by demonstrating that this method is effective across a broader range of algorithms, scales to larger scale tasks, and outperforms two other reward-centering approaches.

## Audience
Those replicating the results in the paper.

Those evaluating new RL algorithms in our testbeds.

Those extending our suite of testbeds to a larger set.

Those seeking deeper insights into deep RL algorithms and reward-centering techniques in continuing tasks.

## Testbeds

Overall we have 21 testbeds, including

- 5 continuous control testbeds without any environment resets, based on five Mujoco environments: Swimmer, HumanoidStandup, Reacher, Pusher, and Ant. The episodic versions of these testbeds are included in Gymnasium (https://github.com/Farama-Foundation/Gymnasium/).  The continuing testbeds are the same as the episodic ones except for the following differences. First, the continuing testbeds do not involve time-based or state-based resets. For Reacher, we resample the target position every 50 steps while leaving the robot's arm untouched, so that the robot needs to learn to reach a new position every 50 steps. Similarly, for Pusher, everything remains the same except that the object's position is randomly sampled every 100 steps. As for Ant, we increase the range of the angles at which its legs can move, so that the ant robot can recover when it flips over.

- 5 continuous control testbeds with predefined environment resets built upon five Mujoco environments: HalfCheetah, Ant, Hopper, Humanoid, and Walker2d.
The corresponding existing episodic testbeds involve time-based truncation of the agent’s experience followed by an environment reset.
In the continuing testbeds, we remove this time-based truncation and reset.
We retain state-based resets, such as when the robot is about to fall (in Hopper, Humanoid, and Walker2d) or when it flips its body (in Ant).
In addition, we add a reset condition for HalfCheetah when it flips, which is not available in the existing episodic testbeds.
Each reset incurs a penalty to the reward, punishing the agent for falling or flipping.

- 6 discrete control testbeds adapted from Atari environments: Breakout, Pong, Space Invaders, BeamRider, Seaquest, and Ms. PacMan. Like the Mujoco environments, the episodic versions include time-based resets, which we omit in the continuing testbeds. In these Atari environments, the agent has multiple lives, and the environment is reset when all lives are lost. Upon losing a life, a reward of -1 is issued as a penalty. Furthermore, in existing algorithmic solutions to episodic Atari testbeds, the rewards are transformed into -1, 0, or 1 by taking their sign for stable learning, though performance is evaluated based on the original rewards. We treat the transformed rewards as the actual rewards in our continuing testbeds, removing such inconsistency.

- 5 Mujoco testbeds with agent-controlled resets, based on five Mujoco environments: HalfCheetah, Ant, Hopper, Humanoid, and Walker2d.
In these testbeds, the agent can choose to reset the environment at any time step.
This is achieved by augmenting the environment's action space in these testbeds by adding one more dimension. This additional dimension has a range of [0, 1], representing the probability of reset.

Because our testbeds are simple modifications of existing episodic Mujoco and Atari testbeds available from [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/), we do not provide a separate package that implements these testbeds.

## Tested algorithms
Continuous control: DDPG, TD3, SAC, PPO

Discrete control: DQN, SAC, PPO

## Results
Here we show the learning curves of tested algorithms in the testbeds, using the hyperparameters that achieve the best overall average reward rate. More results can be found in our paper.

<table>
  <caption style="caption-side: top; font-weight: bold; margin-bottom: 10px;">Mujoco testbeds without resets</caption>
  <tr>
    <td style="text-align: center;"><img src="experiments/no_resets_mujoco/paper/learning_curves_humanoidstandup.png"
    alt="Image 1" width="100"/><br>HumanoidStandup</td>
    <td style="text-align: center;"><img src="experiments/no_resets_mujoco/paper/learning_curves_pushernew.png"
    alt="Image 2" width="100"/><br>Pusher</td>
    <td style="text-align: center;"><img src="experiments/no_resets_mujoco/paper/learning_curves_reachernew.png"
    alt="Image 3" width="100"/><br>Reacher</td>
    <td style="text-align: center;"><img src="experiments/no_resets_mujoco/paper/learning_curves_specialant.png"
    alt="Image 4" width="100"/><br>SpecialAnt</td>
    <td style="text-align: center;"><img src="experiments/no_resets_mujoco/paper/learning_curves_swimmer.png"
    alt="Image 5" width="100"/><br>Swimmer</td>
 </tr>
</table>

<table>
  <caption style="caption-side: top; font-weight: bold; margin-bottom: 10px;">Mujoco testbeds with predefined resets</caption>
  <tr>
    <td style="text-align: center;"><img src="experiments/predefined_resets_mujoco/paper/learning_curves_halfcheetahnew.png" alt="Image 6" width="100"/><br>HalfCheetah</td>
    <td style="text-align: center;"><img src="experiments/predefined_resets_mujoco/paper/learning_curves_antnew.png"
    alt="Image 7" width="100"/><br>Ant</td>
    <td style="text-align: center;"><img src="experiments/predefined_resets_mujoco/paper/learning_curves_hopper.png"
    alt="Image 8" width="100"/><br>Hopper</td>
    <td style="text-align: center;"><img src="experiments/predefined_resets_mujoco/paper/learning_curves_humanoid.png"
    alt="Image 9" width="100"/><br>Humanoid</td>
    <td style="text-align: center;"><img src="experiments/predefined_resets_mujoco/paper/learning_curves_walker2d.png"
    alt="Image 10" width="100"/><br>Walker2d</td>
</tr>
</table>

<table>
  <caption style="caption-side: top; font-weight: bold; margin-bottom: 10px;">Mujoco testbeds with agent-controlled resets</caption>
  <tr>
    <td style="text-align: center;"><img src="experiments/agent_resets_mujoco/paper/learning_curves_halfcheetahnew.png" alt="Image 11" width="100"/><br>HalfCheetah</td>
    <td style="text-align: center;"><img src="experiments/agent_resets_mujoco/paper/learning_curves_antnew.png"
    alt="Image 12" width="100"/><br>Ant</td>
    <td style="text-align: center;"><img src="experiments/agent_resets_mujoco/paper/learning_curves_hopper.png"
    alt="Image 13" width="100"/><br>Hopper</td>
    <td style="text-align: center;"><img src="experiments/agent_resets_mujoco/paper/learning_curves_humanoid.png"
    alt="Image 14" width="100"/><br>Humanoid</td>
    <td style="text-align: center;"><img src="experiments/agent_resets_mujoco/paper/learning_curves_walker2d.png"
    alt="Image 15" width="100"/><br>Walker2d</td>
</tr>
</table>

<table>
  <caption style="caption-side: top; font-weight: bold; margin-bottom: 10px;">Atari testbeds with predefined resets</caption>
  <tr>
    <td style="text-align: center;"><img src="experiments/predefined_resets_atari/paper/learning_curves_breakout.png" alt="Image 16" width="100"/><br>Breakout</td>
    <td style="text-align: center;"><img src="experiments/predefined_resets_atari/paper/learning_curves_beamrider.png"
    alt="Image 17" width="100"/><br>BeamRider</td>
    <td style="text-align: center;"><img src="experiments/predefined_resets_atari/paper/learning_curves_mspacman.png"
    alt="Image 18" width="100"/><br>MsPacman</td>
    <td style="text-align: center;"><img src="experiments/predefined_resets_atari/paper/learning_curves_pong.png"
    alt="Image 19" width="100"/><br>Pong</td>
    <td style="text-align: center;"><img src="experiments/predefined_resets_atari/paper/learning_curves_seaquest.png"
    alt="Image 20" width="100"/><br>Seaquest</td>
    <td style="text-align: center;"><img src="experiments/predefined_resets_atari/paper/learning_curves_spaceinvader.png"
    alt="Image 21" width="100"/><br>SpaceInvader</td>
</tr>
</table>

<table>
  <caption style="caption-side: top; font-weight: bold; margin-bottom: 10px;">Performance improvement when applying reward centering to the tested algorithms.</caption>
  <tr>
    <td style="text-align: center;"><img src="experiments/rc_mujoco/rc_mujoco.png" alt="Image 16" width="500"/><br>Mujoco Testbeds</td>
</tr>
<tr>
    <td style="text-align: center;"><img src="experiments/rc_atari/rc_atari.png" alt="Image 16" width="500"/><br>Atari Testbeds</td>
</tr>
</table>

## How to use the codebase

### Setup conda environment and dependencies
- Create a conda environment ```conda create --name pearl python==3.10```
- ```conda activate pearl```
- Install pearl dependencies using ```./setup.sh```
- For mujoco games, copy ```user_envs/special_ant.xml``` to ```CONDA_DIR_PATH/envs/pearl/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets/```. This is the xml file of the Ant task with a wider range of the angles at which its legs can move. Replace ```CONDA_DIR_PATH``` by the path to the conda directory in your machine.
- For Atari games, we have to manually increase the default maximum episode length in ```CONDA_DIR_PATH/envs/pearl/lib/python3.10/site-packages/ale_py/registration.py```. The default is 108000. You may change it to any number that is larger than the training steps so that the maximum episode length is not reached during training.

### Experiment configurations
The codebase has several experiment folders, each of which includes a file ```inputs.json```, which specifies a set of experiment configurations. This configuration file is compatible with AlphaEx's sweeper for configuration sweeping. https://github.com/AmiiThinks/AlphaEx?tab=readme-ov-file#sweeper explains how to understand the configuration file. Running experiments given these configurations gives experiment results. The table below shows the correspondence between these folders and the figures/tables in the paper summarizing the experiment results.

| Experiment folder | Description | Figures/Tables in the paper |
|---------------|---------------------------|-------------------------|
| ```experiments/no_resets_mujoco/``` | mujoco tasks without resets | Figure 1 (row 1), Table 1, Figure 2,   |
| ```experiments/predefined_resets_mujoco/``` | mujoco tasks with predefined resets | Figure 1 (row 2), Tables 2, 3   |
| ```experiments/agent_resets_mujoco/``` | mujoco tasks with agent resets | Figure 1 (row 3), Table 4, Table 14   |
| ```experiments/rc_mujoco/``` | reward centered algorithms in mujoco tasks without resets or with predefined resets | Table 5 (first two groups), Table 15 (first two groups), Table 17 (first two groups), Table 17 (first two groups), Figures 4, 5   |
| ```experiments/rc_mujoco_agent_reset/``` | reward centered algorithms in Mujoco tasks with agent controlled resets | Table 5 (third group), Table 15 (third groups), Table 17 (third group), Figure 6   |
| ```experiments/rc_mujoco_offset/``` | reward centered algorithms in Mujoco tasks without resets or with predefined resets with reward offsets | Table 16 (first two groups)   |
| ```experiments/rc_mujoco_agent_reset_offset/``` | reward centered algorithms in Mujoco tasks with agent-controlled resets with reward offsets | Table 16 (third group)  |
| ```experiments/predefined_resets_atari/``` | atari tasks with predefined resets | Figure 3, Table 13   |
| ```experiments/rc_atari/``` | reward centered algorithms in atari tasks | Table 6, Table 18, Figure 7 |

### Running experiments
- Suppose we want to perform all experiments specified in ```experiments/no_resets_mujoco/inputs.json``` for ten runs. Note that there are overall 168 experiment configurations in ```experiments/no_resets_mujoco/inputs.json```. Therefore, overall there will be 168 * 10 = 1680 experiments. One could run these experiments sequentially using ```for i in {0..1679}; do ./run.sh run.py --config-file experiments/no_resets_mujoco/inputs.json --out-dir=experiments/no_resets_mujoco --base-id=i; done```. Alternatively, one could run them in parallel using tools like Slurm, depending on the infrastructure available.

### Post-processing experiment results and generating figures and tables
- The first step of post-processing experiment results is to evaluate the final learned policies in predefined_resets_mujoco and predefined_resets_atari experiments. Run ```for i in {0..1599}; do run.sh run.py --config-file experiments/predefined_resets_mujoco/inputs.json    --out-dir=experiments/predefined_resets_mujoco --base-id=i --eval-agent``` and ```for i in {0..719}; do run.sh run.py --config-file experiments/predefined_resets_atari/inputs.json    --out-dir=experiments/predefined_resets_atari --base-id=i --eval-agent```.

- Then generate learning curves and latex code of tables shown in the paper using ```./run_post_processing.sh```.

## Cite us

@article{wan2024continuingtasks,
 title = {An Empirical Study of Deep Reinforcement Learning in Continuing Tasks},
 author = {Yi Wan, Dmytro Korenkevych, Zheqing Zhu},
 year = {2024}
}

## License
Pearl is MIT licensed, as found in the LICENSE file.
