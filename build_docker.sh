docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg INSTALL_BNB=true \
    --build-arg INSTALL_VLLM=true \
    --build-arg INSTALL_DEEPSPEED=true \
    --build-arg INSTALL_FLASHATTN=true \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    -t llmfactory:latest .
