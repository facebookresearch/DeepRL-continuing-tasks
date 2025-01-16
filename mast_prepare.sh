#!/bin/bash

# # A script for running Pearl on a conda Python environment as set up by a companion script.
# # It is meant to run from the root of the pearl source code tree, but makes a copy into
# # an execution directory off the user's $HOME directory
# # because running it inside Meta's repository causes module search try to import
# # Meta's source code for other packages rather than from the Python environment.
# # This copy also has the additional benefit that changes to the source code do not
# # interfere with ongoing executions.

# echo "Activating conda environment mast"
# source /etc/bashrc
# conda activate mast

# # Check if the current directory is "pearl" and contains at least two well-known subdirectories
if [[ "$(pwd)" != *"pearl" ]]; then
    echo "The current directory is not 'pearl'."
    exit 1
fi
if [[ ! -d "api" ]]; then
    echo "The current directory is 'pearl' but it does not contain a subdirectory 'api'. Are you sure this is the right place?"
    exit 1
fi
if [[ ! -d "policy_learners" ]]; then
    echo "The current directory is 'pearl' but it does not contain a subdirectory 'policy_learners'. Are you sure this is the right place?"
    exit 1
fi

# # Directory paths
pearl_dir="$(pwd)"
execution_sandbox_dir="$HOME/pearl_execution"

# # Create the execution directory if it doesn't exist
if [ ! -d "$execution_sandbox_dir" ]; then
  mkdir -p "$execution_sandbox_dir"
fi

# # Clean copy: delete the existing contents of the execution directory
# rm -rf "${execution_sandbox_dir:?}"/*

# # Copy the source directory to the execution directory
cp -r "$pearl_dir" "$execution_sandbox_dir/"

# Enter execution directory
cd "$execution_sandbox_dir" || exit
cp pearl/.torchxconfig .
cp pearl/*.sh .
chmod +777 *.sh
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl
# Executes the given Python file
# script_filename="$1"
# shift
# python "pearl/$script_filename" "$@"
