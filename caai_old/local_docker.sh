#!/bin/bash
docker run --ipc=host --network host --gpus all -v /cm/shared/data/LLaMA-Factory:/workspace -v /cm/shared/data/basemodels:/workspace/basemodels -v /cm/shared/data/outputmodels:/workspace/outputmodels -it llmfactory-test "/bin/bash"
