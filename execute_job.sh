#!/bin/bash

export CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1

cd /app
clearml-agent execute --id $1