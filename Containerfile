FROM docker.io/kyuz0/amd-r9700-toolboxes:rocm-7.2.3

ARG TRELLIS_REPO=https://github.com/Cardboard-box-a/TRELLIS.2_rocm.git
ARG TRELLIS_BRANCH=rocm

LABEL org.opencontainers.image.source="https://github.com/CS6/-r9700-trellis2-rocm-toolbox"
LABEL ai.lab.source="kyuz0/amd-r9700-ai-toolboxes"
LABEL ai.lab.purpose="TRELLIS.2_ROCm_R9700_gfx1201_toolbox"

ENV ROCM_PATH=/opt/rocm-7.2.3
ENV PATH=/opt/rocm/bin:/opt/rocm-7.2.3/bin:/usr/local/bin:/usr/bin
ENV PYTORCH_ROCM_ARCH=gfx1201
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
ENV ROCM_SAFE_SPCONV=1
ENV HF_HOME=/models/huggingface
ENV HUGGINGFACE_HUB_CACHE=/models/huggingface/hub
ENV XDG_CACHE_HOME=/models/cache

SHELL ["/bin/bash", "-lc"]

RUN dnf -y install \
      git \
      python3.12 \
      python3.12-devel \
      python3-pip \
      gcc \
      gcc-c++ \
      make \
      cmake \
      ninja-build \
      patch \
      perl \
      opencv \
      mesa-libGL \
    && dnf clean all

RUN mkdir -p /workspace /models /opt/r9700-trellis2

RUN python3.12 -m venv /workspace/.venv

RUN source /workspace/.venv/bin/activate \
    && python -m pip install --upgrade pip setuptools wheel \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2 \
    && pip install scipy

RUN git clone -b "${TRELLIS_BRANCH}" --recursive "${TRELLIS_REPO}" /workspace/TRELLIS.2_rocm

RUN source /workspace/.venv/bin/activate \
    && cd /workspace/TRELLIS.2_rocm \
    && . ./setup.sh --basic --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec

COPY scripts /opt/r9700-trellis2/scripts
COPY patches /opt/r9700-trellis2/patches

RUN chmod +x /opt/r9700-trellis2/scripts/*.sh \
    && source /workspace/.venv/bin/activate \
    && /opt/r9700-trellis2/scripts/apply-runtime-patches.sh /workspace/TRELLIS.2_rocm /workspace/.venv

WORKDIR /workspace/TRELLIS.2_rocm

CMD ["/bin/bash"]
