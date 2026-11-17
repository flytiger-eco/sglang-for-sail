# 长序列 Prefill Only CP+EP 开启 MegaMoE（GLM-5.3 MXFP4，in100k out1）

本节为《GLM-5.3 推理最佳实践》「MXFP4 长序列 Prefill Only CP+EP（in100k out1）」的补充：在sglang 0.5.21上开启 MegaMoE 的配置方法

## 环境准备2

| 硬件配置 | 真武 890P 单机4卡 |
|------------|----------------------|
| 软件配置 | pkg.flytiger-eco.com/docker_tmp/llm:pytorch2.13.0-ubuntu24.04-cuda13.0-sglang0.5.21-py312-ppu202609291857 |
| 模型权重 | GLM-5.3-MoE-Quant-W-MXFP4-A-MXFP4-Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken |

容器内装包：
```shell
# 1) SGLang 0.5.21（base v0.5.21 + 7 个 open PR，实机验证交付包，测了部分精度，ifbench没测完）
# https://github.com/flytiger-eco/sglang-for-sail/commit/665352faf98b21b09621e5e7bc78aafbea6aaecd
pip install https://pkg.flytiger-eco.com/artifactory/pypi_tmp/2026-11-16-2014/sglang-0.0.0.dev19247+g665352faf.d20261116-1g665352faf98b-cp312-cp312-linux_x86_64.whl --force-reinstall --no-deps

# 在上面的版本的基础上另外还有 5 个 open PR，新编译完全未测试（可选安装这个版本替代上面的版本）
# https://github.com/flytiger-eco/sglang-for-sail/commit/5ade84608bb88d0a5c9b190fe3d074a3563269bd
# pip install https://pkg.flytiger-eco.com/artifactory/pypi_tmp/2026-11-16-2113/sglang-0.0.0.dev19252+g5ade84608.d20261116-1g5ade84608bb8-cp312-cp312-linux_x86_64.whl

# 2) DeepGEMM
pip install https://pkg.flytiger-eco.com/artifactory/pypi_tmp/2026-11-09-1445/deep_gemm-1.1.0.dev15+v1.0.0.ppu2.2.0.g8ce3962-cp312-cp312-linux_x86_64.whl --force-reinstall --no-deps

# 3) FlashMLA
pip install https://pkg.flytiger-eco.com/artifactory/pypi_tmp/2026-11-09-1445/flash_mla-2.1.0.dev105+v1.0.0.ppu2.2.0.g4305d1bd-cp312-cp312-linux_x86_64.whl --force-reinstall --no-deps

# 校验版本
python3 -c "import sglang; print(sglang.__version__)"                       # 0.5.21
python3 -c "import deep_gemm, flash_mla; print('deep_gemm / flash_mla ok')"
```



## server 命令

### **Prefill Only**
```shell
# 进到容器(示例命令如下)
sudo docker run --privileged -it -e HOST_HOSTNAME=$(hostname) \
    --network=host --device=/dev/infiniband --ipc=host --device=/dev/alixpu_ctl --device=/dev/alixpu \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    --init --pid=host \
    -v <path to your workspace> \
    -w <path to your workspace> \
    --name <your docker_name> \
    <PPU DOCKER NAME>

# 关掉代理
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# optional 设置deepgemm cache
CACHE_DIR=/path/to/dg_cache/
export SGLANG_DG_CACHE_DIR=${CACHE_DIR}
export DG_JIT_CACHE_DIR=${SGLANG_DG_CACHE_DIR}
export DG_CACHE_DIR=${SGLANG_DG_CACHE_DIR}

# 本包叠加态必需（与下方 --weight-loader-disable-mmap 配套）
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# 单机4卡【推荐】
# Attn CP + MoE EP，开启 megamoe
python3 -m sglang.launch_server \
  --model-path <GLM-5.3-MXFP4-权重路径> \
  --host 127.0.0.1 --port 31010 \
  --tp-size 4 --ep-size 4 --attn-cp-size 4 --enable-prefill-cp --cp-strategy interleave \
  --chunked-prefill-size 32768 --mem-fraction-static 0.9 \
  --attention-backend dsa --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv \
  --moe-a2a-backend megamoe --enable-w4a4-mxfp4-megamoe \
  --disable-custom-all-reduce --disable-cuda-graph --disable-overlap-schedule \
  --cuda-graph-max-bs-decode 16 --max-running-requests 16 --num-continuous-decode-steps 1 \
  --disable-radix-cache --trust-remote-code --disable-shared-experts-fusion --enable-metrics \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --watchdog-timeout 24000 --dist-timeout 24000 \
  --weight-loader-disable-mmap
```

关闭 megamoe 的对照配置与上述相同，仅把两枚开关替换为 `--moe-a2a-backend deepep --deepep-mode auto`。

## client 命令
```bash
# In 100000 / Out 1，Concurrency 1
python3 -m sglang.bench_serving --max-concurrency 1 --dataset-name random --random-input-len=100000 --random-output-len=1 --random-range-ratio 1 --num-prompts 10 --host 127.0.0.1 --port 31010
```

## 参考性能

| 并行配置 | 节点-卡数 | In / Out | Concurrency | Mean TTFT (ms) | Mean TPOT (ms) | 单卡 Total Throughput (tok/s) | 单卡 Output Throughput (tok/s) |
|------------|-------------|--------|-----------|--------------|--------------|-------------------------------|--------------------------------|
| EP4 TP4 CP4（开 megamoe） | 1-4 | 100000 / 1 | 1 | 9216.67 | — | 2712.13 | 0.03 |
| EP4 TP4 CP4（关 deepep 对照） | 1-4 | 100000 / 1 | 1 | 9640.83 | — | 2592.84 | 0.03 |



## PD 分离配置（P_Node 开启 MegaMoE，890P MXFP4 1P1D）

### P_Node（Prefill，开启 megamoe）
```shell
# 进到容器(示例命令如下)
sudo docker run --privileged -it -e HOST_HOSTNAME=$(hostname) \
    --network=host --device=/dev/infiniband --ipc=host --device=/dev/alixpu_ctl --device=/dev/alixpu \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    --init --pid=host \
    -v <path to your workspace> \
    -w <path to your workspace> \
    --name <your docker_name> \
    <PPU DOCKER NAME>

export SGLANG_DG_CACHE_DIR="/mnt/dg_cache0"
export DG_JIT_CACHE_DIR="/mnt/dg_cache0"
export DG_CACHE_DIR="/mnt/dg_cache0"
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

# 本包叠加态必需（megamoe 相关，与环境章一致）
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# PR #224：PD+CP 前缀缓存开关（默认关；如需开启见章末「如何开启 radix cache」小节，需同时去掉 --disable-radix-cache）
# export SGLANG_DSA_CP_PREFIX_CACHE=1

# Attn CP + MoE EP，开启 megamoe
export DG_USE_MOE_DYNAMIC_TILE=1
export MC_LOG_LEVEL=TRACE
export MC_NUM_QP_PER_EP=4
export MC_TE_METRIC=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_SAIL_DEEPEP_RECV_HOOK=0
python3 -m sglang.launch_server --host 0.0.0.0 --port 8999 --disable-radix-cache --trust-remote-code \
  --model-path <YOUR_MODEL_PATH> --log-level info --enable-cache-report --enable-metrics \
  --dist-timeout 60000 --watchdog-timeout 60000 \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 --tp-size 8 \
  --page-size 64 --speculative-algorithm EAGLE --speculative-attention-mode decode \
  --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
  --disable-custom-all-reduce --moe-a2a-backend megamoe --enable-w4a4-mxfp4-megamoe \
  --disable-shared-experts-fusion --reasoning-parser glm45 --tool-call-parser glm47 \
  --disaggregation-mode prefill --mem-fraction-static 0.85 \
  --attention-backend dsa --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv \
  --enable-prefill-cp --cp-strategy interleave \
  --chunked-prefill-size 32768 --attn-cp-size 8 --disable-overlap-schedule --ep-size 8 \
  --enable-request-time-stats-logging --disable-cuda-graph \
  --weight-loader-disable-mmap
```

### D_Node EP（context 100k）
```shell
sudo docker run --privileged -it -e HOST_HOSTNAME=$(hostname) \
    --network=host --device=/dev/infiniband --ipc=host --device=/dev/alixpu_ctl --device=/dev/alixpu \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    --init --pid=host \
    -v <path to your workspace> \
    -w <path to your workspace> \
    --name <your docker_name> \
    <PPU DOCKER NAME>

export SGLANG_DG_CACHE_DIR="/mnt/dg_cache0"
export DG_JIT_CACHE_DIR="/mnt/dg_cache0"
export DG_CACHE_DIR="/mnt/dg_cache0"
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

# 本包叠加态必需（与 P_Node/单机章一致，缺失会触发全量 precompile warmup 数小时）
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# Attn TP + MoE EP（照参考，deepep 不变）
export DG_USE_MOE_DYNAMIC_TILE=1
export MC_LOG_LEVEL=TRACE
export MC_NUM_QP_PER_EP=4
export MC_TE_METRIC=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_SAIL_DEEPEP_RECV_HOOK=0
python3 -m sglang.launch_server --host 0.0.0.0 --port 8999 --disable-radix-cache --trust-remote-code \
  --model-path <YOUR_MODEL_PATH> --log-level info --enable-cache-report --enable-metrics \
  --dist-timeout 60000 --watchdog-timeout 60000 \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 --tp-size 8 \
  --page-size 64 --speculative-algorithm EAGLE --speculative-attention-mode decode \
  --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
  --disable-custom-all-reduce --moe-a2a-backend deepep --deepep-mode auto \
  --disable-shared-experts-fusion --reasoning-parser glm45 --tool-call-parser glm47 \
  --disaggregation-mode decode --mem-fraction-static 0.8 \
  --prefill-attention-backend fa3 --decode-attention-backend flashmla --cuda-graph-max-bs-decode 32 \
  --max-running-requests 1280 --enable-dp-lm-head --enable-dp-attention --stream-interval 64 \
  --dp-size 8 \
  --weight-loader-disable-mmap
```

### D_Node DP + EP（context 4k）
```shell
sudo docker run --privileged -it -e HOST_HOSTNAME=$(hostname) \
    --network=host --device=/dev/infiniband --ipc=host --device=/dev/alixpu_ctl --device=/dev/alixpu \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    --init --pid=host \
    -v <path to your workspace> \
    -w <path to your workspace> \
    --name <your docker_name> \
    <PPU DOCKER NAME>

export SGLANG_DG_CACHE_DIR="/mnt/dg_cache0"
export DG_JIT_CACHE_DIR="/mnt/dg_cache0"
export DG_CACHE_DIR="/mnt/dg_cache0"
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

# 本包叠加态必需（与 P_Node/单机章一致，缺失会触发全量 precompile warmup 数小时）
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# Attn DP + MoE EP（照参考，deepep 不变）
export DG_USE_MOE_DYNAMIC_TILE=1
export MC_LOG_LEVEL=TRACE
export MC_NUM_QP_PER_EP=4
export MC_TE_METRIC=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_SAIL_DEEPEP_RECV_HOOK=0
python3 -m sglang.launch_server --host 0.0.0.0 --port 8999 --disable-radix-cache --trust-remote-code \
  --model-path <YOUR_MODEL_PATH> --log-level info --enable-cache-report --enable-metrics \
  --dist-timeout 60000 --watchdog-timeout 60000 \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 --tp-size 8 \
  --page-size 64 --speculative-algorithm EAGLE --speculative-attention-mode decode \
  --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
  --disable-custom-all-reduce --moe-a2a-backend deepep --deepep-mode auto \
  --disable-shared-experts-fusion --reasoning-parser glm45 --tool-call-parser glm47 \
  --disaggregation-mode decode --mem-fraction-static 0.8 \
  --prefill-attention-backend fa3 --decode-attention-backend flashmla --cuda-graph-max-bs-decode 32 \
  --chunked-prefill-size 2048 --max-running-requests 1280 --enable-dp-lm-head --enable-dp-attention \
  --stream-interval 64 --dp-size 8 \
  --weight-loader-disable-mmap
```

### Router（on P_node）
```shell
export PREFILL_NODE=<prefill-ip>
export DECODE_NODE=<decode-ip>

#router运行在Prefill相同的Node上 (可以选择不相同的Node)
#router服务需要指明Prefill Node和Decode Node的服务接口
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --host 0.0.0.0 \
  --mini-lb \
  --port 7000 \
  --prefill "http://<prefill-ip>:8999" \
  --decode "http://<decode-ip>:8999"
```

硬口径注（router 地址必须可路由）：--prefill/--decode 必须填对端真实可路由 IP（如 http://<prefill-ip>:8999），不能填 127.0.0.1 或 localhost；填了回环地址时 D 节点会把 router 当 prefill 去握手，PD 间 KV 传输根本建立不起来，但 bench 仍把 HTTP 200＋finish_reason=abort 计入 Successful requests，表面全成功、数据全假。识别法：P 节点日志 ReqTimeStats 中 transfer_total>0 且无 AbortReq/handshake failed；D 日志无「Error fetching prefill server info from bootstrap … Connection refused」。

### client
```bash
DATASET_PATH=<YOUR_LOCAL_DATASET_PATH>

# In 4096 / Out 1500，Concurrency 8
python3 -m sglang.bench_serving --max-concurrency 8 --dataset-name random --random-input-len=4096 --random-output-len=1500 --random-range-ratio 1 --num-prompts 80 --dataset-path ${DATASET_PATH} --host 0.0.0.0 --port 7000

# In 100000 / Out 1000，Concurrency 1
python3 -m sglang.bench_serving --max-concurrency 1 --dataset-name random --random-input-len=100000 --random-output-len=1000 --random-range-ratio 1 --num-prompts 10 --dataset-path ${DATASET_PATH} --host 0.0.0.0 --port 7000
```

口径注（PD 压测合格判据）：PD 链路可能静默失败——router 地址填错（见 Router 小节硬口径注）时 KV 传输建立不起来，bench 仍把 abort 请求计入 Successful requests。识别特征：Total generated tokens (retokenized): 0 且每请求时延≈14s（bootstrap 15 次重试常数）。合格判据：P 日志 ReqTimeStats transfer_total>0 且无 AbortReq/handshake failed，retokenized 非零。链路健康时 bench 的 TTFT/TPOT 可信（本包已用 /generate stream=true 直测交叉验证，差约1%）。

### 参考性能

| Prefill 并行 | Decode 并行 | 节点-卡数 | In / Out | Concurrency | Mean TTFT (ms) | Mean TPOT (ms) | 单卡 Total Throughput (tok/s) | 单卡 Output Throughput (tok/s) |
|--------------|-------------|-------------|--------|-----------|--------------|--------------|-------------------------------|--------------------------------|
| CP8 TP8 EP8（开 megamoe） | EP8 TP1 DP8 | 2-8 | 4096 / 1500 | 8 | 790.48 | 13.52 | 131.59 | 35.27 |
| CP8 TP8 EP8（开 megamoe） | EP8 TP1 DP8 | 2-8 | 100000 / 1000 | 1 | 24723.00 | 17.85 | 148.33 | 1.47 |

表注：
- 数据为 1P1D 两台×8卡共16卡（P 开 megamoe、D deepep），单卡＝总量÷16；两侧均开 MTP（EAGLE）。
- TTFT/TPOT 为 bench_serving 实测，已与 /generate 流式直测交叉验证（100k 稳态 TTFT 直测中位 4850.62ms，与 bench 中位 4904.11ms 差约1%）。
- 100k case 的 Mean TTFT 受 2 个 prefill 长尾请求拉高（中位 4904.11ms），长尾根因待后续单请求取证，如实标注。

### 如何开启 radix cache（前缀缓存）
- 配置方法：P 节点设置环境变量 SGLANG_DSA_CP_PREFIX_CACHE=1，且启动命令不带 --disable-radix-cache。两件事必须一起做：该旗标是显式关闭，带着它时环境变量无效。
- 生效条件（缺一不可）：PD 分离 prefill 节点 ＋ --enable-prefill-cp ＋ --cp-strategy interleave ＋ --page-size 64（page_size≠64 会直接拒启）。
- 生效判据：server.log 出现 "DSA prefill CP prefix cache is ENABLED (experimental)"；若仍出现 "Radix cache is disabled on this PD worker because DSA prefill CP requires…" 说明未生效。
- 不开启时：保持默认（不设该变量）即可，行为与之前一致。D 节点不适用此开关。


