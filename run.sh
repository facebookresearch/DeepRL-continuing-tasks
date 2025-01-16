#!/usr/bin/bash
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# cd $SCRIPT_DIR &&
if [ "$LOCAL_RANK" = "0" ] && [ -z "$DISABLE_MOUNT" ]; then
    source /packages/torchx_conda_mount/mount.sh
fi
export PYTHONPATH=.
export MUJOCO_GL=disable
python3 "$@"
