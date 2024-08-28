#!/bin/bash

cd /app

export UUID=$(uuidgen)
export WORKING_DIR=/tmp/working_dir/$UUID
export HOME=$WORKING_DIR

echo "WORKING_DIR: $WORKING_DIR"

export RANK=${PMIX_RANK}

echo "RANK: $RANK"

export CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1
export CLEARML_VENVS_BUILDS=$WORKING_DIR/.clearml/venvs-builds
export CLEARML_VCS_CACHE=$WORKING_DIR/.clearml/vcs-cache
export CLEARML_PIP_CACHE=$WORKING_DIR/.clearml/pip-download-cache
export CLEARML_DOCKER_PIP_CACHE=$WORKING_DIR/.clearml/pip-cache
export CLEARML_APT_CACHE=$WORKING_DIR/.clearml/apt-cache
export CLEARML_TASK_NO_REUSE="1"
export CLEARML_LOG_LEVEL="INFO"

mkdir -p $WORKING_DIR

clearml-agent execute --id $1

rm -rf $WORKING_DIR


