FROM mambaorg/micromamba:1.5.6 as micromamba

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS build

WORKDIR /build

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS "-Xfatbin -compress-all"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    build-essential \
    curl \
    git \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ENV PATH=/opt/conda/bin:$PATH

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_setup_root_prefix.sh

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
CMD ["/bin/bash"]

RUN micromamba install -y -n base -c pytorch -c nvidia -c conda-forge python=3.10 pytorch=2.0.1 pytorch-cuda=11.8 mkl=2024.0 numpy && \
    micromamba clean --all --yes

RUN git clone https://github.com/NVIDIA/MinkowskiEngine -q
RUN MAX_JOBS=4 micromamba run -n base python -m pip wheel -v --disable-pip-version-check --no-deps --no-cache-dir \
    --build-option="--force_cuda" \
    --build-option="--blas_include_dirs=${MAMBA_ROOT_PREFIX}/include" \
    --build-option="--blas=mkl" \
    ./MinkowskiEngine


FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 AS runtime

ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ENV PATH=/opt/conda/bin:$PATH

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_setup_root_prefix.sh

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
CMD ["/bin/bash"]
    
ADD environment.yml ./
COPY --from=build /build/*.whl /wheels/
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes && \
    micromamba run -n base pip cache purge && \
    rm environment.yml

WORKDIR /app
COPY *.py ./
COPY configs ./configs
