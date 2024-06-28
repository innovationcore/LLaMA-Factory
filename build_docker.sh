
export MAX_JOBS=40

docker build -f ./Dockerfile_llmfactory \
    --build-arg INSTALL_BNB=false \
    --build-arg INSTALL_VLLM=false \
    --build-arg INSTALL_DEEPSPEED=true \
    --build-arg INSTALL_FLASHATTN=true \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    -t llmfactory:latest .
