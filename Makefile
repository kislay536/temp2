# NOTE: slamio-main / slamio-baseline-splatam are NOT needed here. The raw
# upstream run_slamio_final_configs.sh (slamio_final branch) self-provisions
# its own stock rasterizer on first run (ensure_splatam_stock_rasterizer()),
# caching the build under ~/.cache/slamio/splatam-stock-rasterizer -- that
# mechanism only needs a working nvcc (see CUDA_HOME below), not a second env.
build:
	test -d slamio-final || git clone -b slamio_final https://github.com/AggelikiZ/3DGS-framework.git slamio-final
	mkdir -p envs-h100/logs && ENVS_DIR="$(CURDIR)/envs-h100" TORCH_CUDA_ARCH_LIST="9.0" bash scripts/setup_env.sh slamio-opt-splatam

dataset:
	cd slamio-final && bash setup_datasets.sh && bash download_replica.sh room0 && cd ..

REPLICA_SRC := /scratch/sws0/user/karya/3dgs/datasets/replica/Replica

dataset-symlink:
	mkdir -p slamio-final/SplaTAM/data
	ln -sfn "$(REPLICA_SRC)" slamio-final/SplaTAM/data/Replica

# CUDA_HOME points at the GCC-14-compatible "wrapped" nvcc that setup_env.sh
# already built as a side effect of `make build` (at envs-h100/.cuda-wrapped).
# The raw system nvcc (e.g. /usr/lib/cuda-12.1) rejects this host's GCC
# version outright ("gcc versions later than 12 are not supported") when
# ensure_splatam_stock_rasterizer() self-builds the stock rasterizer on
# first run -- confirmed by direct test, the wrapped nvcc compiles clean.
run-replica:
	cd slamio-final && \
	SPLATAM_PYTHON="$(CURDIR)/envs-h100/slamio-opt-splatam/bin/python" \
	CUDA_HOME="$(CURDIR)/envs-h100/.cuda-wrapped" \
	REPS=1 GPU_INDEX=0 COMPUTE_LOCATION=local \
	bash profiling/run_slamio_final_configs.sh run splatam replica

