# Qwen38 EPD image pruning checks

Run these manual checks in the SGLang CUDA image used for Qwen38 EPD (Python 3.12, PyTorch, pytest and the FA3 extension). The tests cover dominant token selection and DeepStack alignment, original MRoPE coordinates and continuation positions, atomic expanded-length metadata, local embedding-cache metadata and out-of-order Encoder transport. Source-extracted protocol tests avoid loading a model; a real tiny vision forward is opt-in.

```bash
PYTHONPATH=python python3 -m pytest -q test/manual/epd_pruning
RUN_VISION_GPU_SMOKE=1 PYTHONPATH=python python3 -m pytest -q test/manual/epd_pruning/test_vision_pruning.py
```

The GPU smoke initializes a one-rank NCCL group and runs a two-layer vision model with two images; use an idle GPU. Full EPD deployment additionally needs text and multi-image requests through the router after all workers and the cohort become ready.
