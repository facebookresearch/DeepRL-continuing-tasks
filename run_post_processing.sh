#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


export PYTHONPATH=.

# create plots and generate results

# mujoco tasks without resets
python plot.py --plot-config-file experiments/refactor/no_resets_mujoco/plot.json > experiments/refactor/no_resets_mujoco/plot.txt &

# mujoco tasks without resets higher exploration
python plot.py --plot-config-file experiments/refactor/no_resets_mujoco_higher_exp_noise/plot.json > experiments/refactor/no_resets_mujoco_higher_exp_noise/plot.txt &

# mujoco tasks with small random reset probability
python plot.py --plot-config-file experiments/refactor/no_resets_mujoco/plot_resets_improvement_table.json > experiments/refactor/no_resets_mujoco/plot_resets_improvement_table.txt &

# mujoco tasks with smaller state space and no resets
python plot.py --plot-config-file experiments/refactor/no_resets_mujoco/plot_reduced_state_space_improvement_table.json > experiments/refactor/no_resets_mujoco/plot_reduced_state_space_improvement_table.txt &

# mujoco tasks with predefined resets
python plot.py --plot-config-file experiments/refactor/predefined_resets_mujoco/plot.json > experiments/refactor/predefined_resets_mujoco/plot.txt &

# returns of mujoco episodic tasks with predefined resets
python plot.py --plot-config-file experiments/refactor/predefined_resets_mujoco/plot_episodic_return.json > experiments/refactor/predefined_resets_mujoco/plot_episodic_return.txt &

# comparing the reward rate and number of resets of learned policies in continuing and episodic mujoco tasks
python plot.py --plot-config-file experiments/refactor/predefined_resets_mujoco/plot_continuing_episodic_comparison_eval_reward_rate.json > experiments/refactor/predefined_resets_mujoco/plot_continuing_episodic_comparison_eval_reward_rate.txt &
python plot.py --plot-config-file experiments/refactor/predefined_resets_mujoco/plot_continuing_episodic_comparison_eval_reward_rate_no_reset_costs.json > experiments/refactor/predefined_resets_mujoco/plot_continuing_episodic_comparison_eval_reward_rate_no_reset_costs.txt &
python plot.py --plot-config-file experiments/refactor/predefined_resets_mujoco/plot_continuing_episodic_comparison_eval_num_resets.json > experiments/refactor/predefined_resets_mujoco/plot_continuing_episodic_comparison_eval_num_resets.txt &

# comparing the reward rate and number of resets of learned policies in continuing mujoco tasks with different reset costs
python plot.py --plot-config-file experiments/refactor/predefined_resets_mujoco/plot_reset_costs_comparison_eval_reward_rate.json > experiments/refactor/predefined_resets_mujoco/plot_reset_costs_comparison_eval_reward_rate.txt &
python plot.py --plot-config-file experiments/refactor/predefined_resets_mujoco/plot_reset_costs_comparison_eval_reward_rate_no_reset_costs.json > experiments/refactor/predefined_resets_mujoco/plot_reset_costs_comparison_eval_reward_rate_no_reset_costs.txt &
python plot.py --plot-config-file experiments/refactor/predefined_resets_mujoco/plot_reset_costs_comparison_eval_num_resets.json > experiments/refactor/predefined_resets_mujoco/plot_reset_costs_comparison_eval_num_resets.txt &

# mujoco tasks with agent-controlled resets
python plot.py --plot-config-file experiments/refactor/agent_resets_mujoco/plot.json > experiments/refactor/agent_resets_mujoco/plot.txt &

# comparing the reward rate and number of resets of learned policies in mujoco tasks with agent-controlled resets and predefined resets
python plot.py --plot-config-file experiments/refactor/agent_resets_mujoco/plot_agent_controlled_predefined_comparison_eval_reward_rate.json > experiments/refactor/agent_resets_mujoco/plot_agent_controlled_predefined_comparison_eval_reward_rate.txt &
python plot.py --plot-config-file experiments/refactor/agent_resets_mujoco/plot_agent_controlled_predefined_comparison_eval_reward_rate_no_reset_costs.json > experiments/refactor/agent_resets_mujoco/plot_agent_controlled_predefined_comparison_eval_reward_rate_no_reset_costs.txt &
python plot.py --plot-config-file experiments/refactor/agent_resets_mujoco/plot_agent_controlled_predefined_comparison_eval_num_resets.json > experiments/refactor/agent_resets_mujoco/plot_agent_controlled_predefined_comparison_eval_num_resets.txt &

# mujoco tasks with reward centering
python plot.py --plot-config-file experiments/refactor/rc_mujoco/plot.json > experiments/refactor/rc_mujoco/plot.txt &

# mujoco tasks with different discount factors
python plot.py --plot-config-file experiments/refactor/rc_mujoco/plot_discount.json > experiments/refactor/rc_mujoco/plot_discount.txt &

# mujoco tasks with different offsets in rewards
python plot.py --plot-config-file experiments/refactor/rc_mujoco_offset/plot.json > experiments/refactor/rc_mujoco_offset/plot.txt &

# atari tasks with predefined resets
python plot.py --plot-config-file experiments/refactor/predefined_resets_atari/plot.json > experiments/refactor/predefined_resets_atari/plot.txt &

# comparing the reward rate and number of resets of learned policies in continuing and episodic atari tasks
python plot.py --plot-config-file experiments/refactor/predefined_resets_atari/plot_continuing_episodic_comparison_eval_reward_rate.json > experiments/refactor/predefined_resets_atari/plot_continuing_episodic_comparison_eval_reward_rate.txt &
python plot.py --plot-config-file experiments/refactor/predefined_resets_atari/plot_continuing_episodic_comparison_eval_reward_rate_no_reset_costs.json > experiments/refactor/predefined_resets_atari/plot_continuing_episodic_comparison_eval_reward_rate_no_reset_costs.txt &
python plot.py --plot-config-file experiments/refactor/predefined_resets_atari/plot_continuing_episodic_comparison_eval_num_resets.json > experiments/refactor/predefined_resets_atari/plot_continuing_episodic_comparison_eval_num_resets.txt &

# atari tasks with reward centering
python plot.py --plot-config-file experiments/refactor/rc_atari/plot.json > experiments/refactor/rc_atari/plot.txt &

wait

# showing the results in latex tables

# mujoco no resets tables
python3 experiments/refactor/no_resets_mujoco/latex_table_resets_improvement_table.py experiments/refactor/no_resets_mujoco/plot_resets_improvement_table.txt > experiments/refactor/no_resets_mujoco/latex_table_resets_improvement.txt &
python3 experiments/refactor/no_resets_mujoco/latex_table_reduced_state_space_improvement_table.py experiments/refactor/no_resets_mujoco/plot_reduced_state_space_improvement_table.txt > experiments/refactor/no_resets_mujoco/latex_table_reduced_state_space_improvement.txt &

# mujoco predefined resets tables
python3 experiments/refactor/predefined_resets_mujoco/latex_table_continuing_episodic_comparison.py experiments/refactor/predefined_resets_mujoco/plot_continuing_episodic_comparison_eval_reward_rate.txt experiments/refactor/predefined_resets_mujoco/plot_continuing_episodic_comparison_eval_num_resets.txt> experiments/refactor/predefined_resets_mujoco/latex_table_continuing_episodic_comparison.txt &
python3 experiments/refactor/predefined_resets_mujoco/latex_table_reset_costs_comparison.py experiments/refactor/predefined_resets_mujoco/plot_reset_costs_comparison_eval_reward_rate_no_reset_costs.txt experiments/refactor/predefined_resets_mujoco/plot_reset_costs_comparison_eval_num_resets.txt > experiments/refactor/predefined_resets_mujoco/latex_table_reset_costs_comparison.txt &

# mujoco agent-controlled resets tables
python3 experiments/refactor/agent_resets_mujoco/latex_table_agent_controlled_predefined_comparison.py experiments/refactor/agent_resets_mujoco/plot_agent_controlled_predefined_comparison_eval_reward_rate.txt experiments/refactor/agent_resets_mujoco/plot_agent_controlled_predefined_comparison_eval_num_resets.txt > experiments/refactor/agent_resets_mujoco/latex_table_agent_controlled_predefined_comparison.txt &

# mujoco reward centering table
python3 experiments/refactor/rc_mujoco/latex_table.py experiments/refactor/rc_mujoco/plot.txt > experiments/refactor/rc_mujoco/latex_table.txt &

# mujoco discount table
python3 experiments/refactor/rc_mujoco/latex_table_discount.py experiments/refactor/rc_mujoco/plot_discount.txt > experiments/refactor/rc_mujoco/latex_table_discount.txt &

# mujoco reward offset table
python3 experiments/refactor/rc_mujoco_offset/latex_table.py experiments/refactor/rc_mujoco_offset/plot.txt > experiments/refactor/rc_mujoco_offset/latex_table.txt &

# atari predefined resets tables
python3 experiments/refactor/predefined_resets_atari/latex_table_continuing_episodic_comparison.py experiments/refactor/predefined_resets_atari/plot_continuing_episodic_comparison_eval_reward_rate.txt experiments/refactor/predefined_resets_atari/plot_continuing_episodic_comparison_eval_num_resets.txt > experiments/refactor/predefined_resets_atari/latex_table_continuing_episodic_comparison.txt &

# atari reward centering table
python3 experiments/refactor/rc_atari/latex_table.py experiments/refactor/rc_atari/plot.txt > experiments/refactor/rc_atari/latex_table.txt &
