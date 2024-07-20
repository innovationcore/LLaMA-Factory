#!/bin/bash

cd /app

export UUID=$(uuidgen)
export WORKING_DIR=/app/working_dir/$UUID
export HOME=$WORKING_DIR

export CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1
export CLEARML_VENVS_BUILDS=$WORKING_DIR/.clearml/venvs-builds
export CLEARML_VCS_CACHE=$WORKING_DIR/.clearml/vcs-cache
export CLEARML_PIP_CACHE=$WORKING_DIR/.clearml/pip-download-cache
export CLEARML_DOCKER_PIP_CACHE=$WORKING_DIR/.clearml/pip-cache
export CLEARML_APT_CACHE=$WORKING_DIR/.clearml/apt-cache
export CLEARML_TASK_NO_REUSE="1"
export CLEARML_LOG_LEVEL="INFO"

mkdir -p $WORKING_DIR

python clearml_training_wrapper.py --dataset_file=b384e92e-8fe6-4f06-960e-cd7eff15cb8f.json --dataset_name=b384e92e-8fe6-4f06-960e-cd7eff15cb8f --task_name=trainer_template_v23

rm -rf $WORKING_DIR
rm -rf $CLEARML_CUSTOM_TASK_DATA_PATH
~
~