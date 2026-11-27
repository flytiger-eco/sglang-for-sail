"""PPU single-launch combine/RMS, down/gate, split-K reduction and mix-up."""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.gemm.hc_down_gate import _pick_config

logger = logging.getLogger(__name__)

# A capture-time fallback is otherwise silent: the graph simply contains the
# split-kernel path instead. Log once per (reason, row count).
_fallback_logged: set = set()


def _log_capture_fallback(reason: str, rows: int) -> None:
    key = (reason, rows)
    if key in _fallback_logged:
        return
    _fallback_logged.add(key)
    logger.warning(
        "gated-HC mega kernel unavailable during CUDA graph capture (%s, rows=%d); "
        "falling back to the split-kernel path for this shape",
        reason,
        rows,
    )


@triton.jit
def _mega_wait(counter, ready, end):
    while ready != end:
        ready = tl.atomic_add(counter, 0, sem="acquire", scope="gpu").to(tl.uint32)


@triton.jit
def _publish_partial(counter, must_wait, CUTS: tl.constexpr):
    mask = tl.full((), 0xffffffff ^ (CUTS - 1), tl.uint32)
    step = tl.full((), CUTS, tl.uint32)
    tl.inline_asm_elementwise(
        "{\n.reg .pred skip, pending;\n.reg .u32 tid, observed, target;\nmov.u32 tid, %tid.x;\nsetp.ne.u32 skip, tid, 0;\n@skip bra part_exit;\natom.acq_rel.gpu.global.add.u32 observed, [$1], 1;\nsetp.eq.u32 skip, $2, 0;\n@skip bra part_exit;\nand.b32 target, observed, $3;\nadd.u32 target, target, $4;\nadd.u32 observed, observed, 1;\nsetp.eq.u32 skip, observed, target;\n@skip bra part_exit;\npart_wait:\nld.acquire.gpu.global.u32 observed, [$1];\nsetp.ne.u32 pending, observed, target;\n@pending bra part_wait;\npart_exit:\nmov.u32 $0, 0;\n}",
        "=r,l,r,r,r", [counter, must_wait.to(tl.uint32), mask, step],
        dtype=tl.uint32, is_pure=False, pack=1)
    if must_wait:
        tl.debug_barrier()


@triton.jit
def _down(X, WD, WG, P, A, G, S, pid, logical, ticket, epoch,
          M: tl.constexpr, BM: tl.constexpr, SK: tl.constexpr,
          DEPTH: tl.constexpr, GROUP: tl.constexpr, X_CACHE: tl.constexpr,
          REQUESTED: tl.constexpr, OWNERS: tl.constexpr, TURN: tl.constexpr,
          STATE_STRIDE: tl.constexpr, FUSE_WAIT: tl.constexpr,
          ACT_LAYOUT: tl.constexpr, DOWN_AIU: tl.constexpr, W_CACHE: tl.constexpr,
          A_EVICT: tl.constexpr, W_EVICT: tl.constexpr):
    tile = logical % 6
    split = logical // 6
    chunk: tl.constexpr = 10240 // SK
    row = tl.arange(0, BM)
    k = tl.arange(0, 64)
    n = tl.arange(0, 64)
    valid_n = tile * 64 + n < 324
    xp = X + row[:, None] * 10240 + split * chunk + k[None, :]
    columns = tl.where(tile < 5, tile * 64 + n, n)
    weight = tl.where(tile < 5, WD, WG) + columns[None, :] * 10240 + split * chunk + k[:, None]
    _mega_wait(S, ticket, epoch)
    acc = tl.zeros((BM, 64), tl.float32)
    for step in tl.range(chunk // 64, num_stages=DEPTH):
        if DOWN_AIU and M == BM:
            if DOWN_AIU == 1:
                block = tl.make_block_ptr(X, (BM, 10240), (10240, 1),
                                          (0, split * chunk + step * 64), (BM, 64), (1, 0))
            else:
                base = X + split * chunk
                block = tl.make_block_ptr(base, (BM, chunk), (10240, 1),
                                          (0, step * 64), (BM, 64), (1, 0))
            a = tl.aiu_load(block)
        else:
            a = tl.load(xp + step * 64, row[:, None] < M, 0,
                        cache_modifier=X_CACHE, eviction_policy=A_EVICT)
        w = tl.load(weight + step * 64, valid_n[None, :], 0,
                    cache_modifier=W_CACHE, eviction_policy=W_EVICT)
        acc = tl.dot(a, w, acc)
    stripe: tl.constexpr = BM // (GROUP if GROUP else 1)
    if GROUP:
        write_plane = tile * GROUP + row // stripe
        packed_row = (write_plane * SK + split) * stripe + row % stripe
    else:
        packed_row = (tile * SK + split) * BM + row
    tl.store(P + packed_row[:, None] * 64 + n[None, :],
             acc, (row[:, None] < M) & valid_n[None, :], cache_modifier=".cg")
    tl.debug_barrier()
    done = S + (1 + tile) * STATE_STRIDE
    reducers: tl.constexpr = min(REQUESTED, SK)
    offset: tl.constexpr = SK - reducers if OWNERS == 0 else TURN
    if OWNERS == 3:
        offset = TURN + tile % 2
    elif OWNERS == 4:
        offset = TURN + tile // 2
    index = (split - offset) & (SK - 1)
    if OWNERS >= 2:
        gap: tl.constexpr = SK // reducers
        active_owner = index % gap == 0
        contributor = index // gap
    else:
        active_owner = index < reducers
        contributor = index
    if FUSE_WAIT:
        _publish_partial(done, active_owner, SK)
    else:
        serial = tl.atomic_add(done, 1, sem="acq_rel", scope="gpu").to(tl.uint32)
    if active_owner:
        if not FUSE_WAIT:
            finished = (serial & (0xffffffff ^ (SK - 1))) + SK
            _mega_wait(done, serial + 1, finished)
        width: tl.constexpr = BM // reducers
        span: tl.constexpr = max(width * 64, 1024)
        items = tl.arange(0, span)
        m = contributor * width + items // 64
        n_local = items % 64
        col = tile * 64 + n_local
        active = (items < width * 64) & (m < M) & (col < 324)
        s = tl.arange(0, SK)
        if GROUP:
            read_plane = tile * GROUP + m // stripe
            packed_m = (read_plane[None, :] * SK + s[:, None]) * stripe + (m % stripe)[None, :]
        else:
            packed_m = (tile * SK + s[:, None]) * BM + m[None, :]
        offsets = packed_m * 64 + n_local[None, :]
        total = tl.sum(tl.load(P + offsets, active[None, :], 0, cache_modifier=".cg"), axis=0)
        if tile < 5:
            rounded = (total.to(tl.bfloat16).to(tl.float32) * 0.25).to(tl.bfloat16).to(tl.float32)
            if ACT_LAYOUT == 1:
                act_offset = (tile * BM + m) * 64 + n_local
            elif ACT_LAYOUT == 2:
                act_offset = ((tile * 2 + n_local // 32) * BM + m) * 32 + n_local % 32
            else:
                act_offset = m * 320 + col
            tl.store(A + act_offset, (rounded * tl.sigmoid(rounded)).to(tl.bfloat16),
                     active, cache_modifier=".cg")
            tl.debug_barrier()
            count = tl.where((tile == 0) & (contributor == 0), 513 - 5 * reducers, 1)
            tl.atomic_add(S + 7 * STATE_STRIDE, count, sem="acq_rel", scope="gpu")
        else:
            tl.store(G + m * 4 + n_local, 2 * tl.sigmoid(total * 0.25), active)


@triton.jit
def _logical_down(pid, SK: tl.constexpr, MAPPING: tl.constexpr, ROTATE: tl.constexpr):
    if MAPPING == 4 and SK == 16:
        lane = pid % 39
        wave = pid // 39
        if lane < 18:
            split = lane // 2
            tile = (lane % 2) * 3 + wave
            logical = ((split + ROTATE) % 16) * 6 + tile
        else:
            if wave < 2:
                split = 9 + (lane - 18) // 3
                tile = ((lane - 18) % 3) * 2 + wave
                logical = ((split + ROTATE) % 16) * 6 + tile
            else:
                logical = 96 + lane - 18
    elif MAPPING == 1:
        logical = (pid % SK) * 6 + pid // SK
        logical = tl.where(pid < SK * 6, logical, pid)
    elif MAPPING == 2 or MAPPING == 4:
        logical = tl.where(pid < 78, (pid % 39) * 2 + pid // 39, pid)
    elif MAPPING == 3:
        logical = (pid % 39) * 3 + pid // 39
    else:
        logical = pid
    return logical


@triton.jit
def _tail_order(pid, CTAS: tl.constexpr, WAVE: tl.constexpr, ROT: tl.constexpr):
    lane = pid % 39
    stage = pid // 39
    if WAVE == 1:
        return tl.where(pid < 39, pid + 39, tl.where(pid < 78, pid - 39, pid))
    if WAVE == 2:
        return (pid + CTAS - 78) % CTAS
    if WAVE == 3:
        return tl.where(stage == 1, 39 + (lane + 39 - ROT) % 39, pid)
    if WAVE == 4:
        return tl.where(stage == 0, 39 + (lane + 39 - ROT) % 39,
                        tl.where(stage == 1, lane, pid))
    if WAVE == 5:
        return tl.where(pid < 78, lane * 2 + stage, pid)
    if WAVE == 6:
        return tl.where(pid >= 78, pid - 78, CTAS - 78 + stage * 39 + (lane + 39 - ROT) % 39)
    return pid


@triton.jit
def _norm_pairs(u, BATCH: tl.constexpr):
    even, odd = tl.split(tl.reshape(u, (BATCH * 256, 4, 2)))
    pairs = even * even + odd * odd
    first, second = tl.split(tl.reshape(pairs, (BATCH * 256, 2, 2)))
    p0, p2 = tl.split(first)
    p1, p3 = tl.split(second)
    return p0, p1, p2, p3


@triton.jit
def _warp_norm(B, RES, OLD_G, W, U, X, task, BATCH: tl.constexpr,
               SPREAD: tl.constexpr, GATE_PURE: tl.constexpr,
               WEIGHT_WHEN: tl.constexpr, WEIGHT_CACHE: tl.constexpr):
    rows: tl.constexpr = BATCH * 8
    rounds: tl.constexpr = 8 // SPREAD
    group = tl.arange(0, rows)
    branch = (group // SPREAD) % BATCH
    logical_warp = (group // (BATCH * SPREAD)) * SPREAD + group % SPREAD
    segment = logical_warp[:, None] * 256 + tl.arange(0, 256)[None, :]
    valid = logical_warp[:, None] < 5
    tasks = task * BATCH + branch
    offsets = tasks[:, None] * 2560 + segment
    base = B + (task // (4 // BATCH)) * 2560
    wt = W + (tasks % 4)[:, None] * 2560 + segment
    if WEIGHT_WHEN == 3:
        w0 = tl.load(wt, valid, 0, cache_modifier=WEIGHT_CACHE)
        w1 = tl.load(wt + 1280, valid, 0, cache_modifier=WEIGHT_CACHE)
    if GATE_PURE:
        bits = tl.inline_asm_elementwise("ld.global.b32 $0, [$1];", "=r,l", [OLD_G + tasks],
                                         dtype=tl.uint32, is_pure=True, pack=1)
        gates = bits.to(tl.float32, bitcast=True)
    else:
        gates = tl.load(OLD_G + tasks)
    r0 = tl.load(RES + offsets, valid, 0).to(tl.float32)
    r1 = tl.load(RES + offsets + 1280, valid, 0).to(tl.float32)
    b0 = tl.load(base + segment, valid, 0).to(tl.float32)
    b1 = tl.load(base + segment + 1280, valid, 0).to(tl.float32)
    u0 = tl.fma(b0, gates[:, None], r0).to(tl.bfloat16)
    u1 = tl.fma(b1, gates[:, None], r1).to(tl.bfloat16)
    tl.store(U + offsets, u0, valid, cache_modifier=".cg")
    tl.store(U + offsets + 1280, u1, valid, cache_modifier=".cg")
    u0_f = u0.to(tl.float32)
    u1_f = u1.to(tl.float32)
    if WEIGHT_WHEN == 2:
        w0 = tl.load(wt, valid, 0, cache_modifier=WEIGHT_CACHE)
        w1 = tl.load(wt + 1280, valid, 0, cache_modifier=WEIGHT_CACHE)
    p0, p1, p2, p3 = _norm_pairs(u0_f, BATCH)
    p4, p5, p6, p7 = _norm_pairs(u1_f, BATCH)
    sums = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7
    warp_sums = tl.sum(tl.reshape(sums, (rows, 32)), axis=1)
    phases = tl.reshape(warp_sums, (rounds, BATCH, SPREAD))
    warped = tl.reshape(tl.permute(phases, (1, 0, 2)), (BATCH, 8))
    for power in tl.static_range(2, -1, -1):
        piece = tl.reshape(warped, (BATCH, 2, 1 << power))
        low, high = tl.split(tl.permute(piece, (0, 2, 1)))
        warped = low + high
    total = tl.reshape(warped, (BATCH,))
    if WEIGHT_WHEN == 1:
        w0 = tl.load(wt, valid, 0, cache_modifier=WEIGHT_CACHE)
        w1 = tl.load(wt + 1280, valid, 0, cache_modifier=WEIGHT_CACHE)
    q = total * (1.0 / 2560)
    e = tl.fma(-q, 2560.0, total)
    mean = tl.fma(e, 1.0 / 2560, q)
    inv = tl.rsqrt(mean + 1e-6)
    by_group = tl.broadcast_to(inv[None, :, None], (rounds, BATCH, SPREAD))
    by_group = tl.reshape(by_group, (rows, 1))
    if WEIGHT_WHEN == 0:
        w0 = tl.load(wt, valid, 0, cache_modifier=WEIGHT_CACHE)
        w1 = tl.load(wt + 1280, valid, 0, cache_modifier=WEIGHT_CACHE)
    x0 = (u0_f * by_group * (1.0 + w0.to(tl.float32))).to(tl.bfloat16)
    x1 = (u1_f * by_group * (1.0 + w1.to(tl.float32))).to(tl.bfloat16)
    tl.store(X + offsets, x0, valid, cache_modifier=".cg")
    tl.store(X + offsets + 1280, x1, valid, cache_modifier=".cg")


@triton.jit
def _norm_tail(B, R, G0, WN, U, X, pid, M: tl.constexpr, CTAS: tl.constexpr,
               BATCH: tl.constexpr, SPREAD: tl.constexpr, BACK: tl.constexpr,
               WAVE: tl.constexpr, ROT: tl.constexpr, BRANCH_MAJOR: tl.constexpr,
               WEIGHT_WHEN: tl.constexpr, WEIGHT_CACHE: tl.constexpr):
    if M > CTAS:
        items: tl.constexpr = (M - CTAS) * (4 // BATCH)
        tl.static_assert(items <= CTAS, "RMS tail can assign one item per CTA")
        start: tl.constexpr = CTAS - items if BACK else 0
        rank = _tail_order(pid, CTAS, WAVE, ROT) - start
        if (rank >= 0) & (rank < items):
            if BRANCH_MAJOR:
                tail_rows: tl.constexpr = M - CTAS
                rank = (rank % tail_rows) * (4 // BATCH) + rank // tail_rows
            task = CTAS * (4 // BATCH) + rank
            _warp_norm(B, R, G0, WN, U, X, task, BATCH, SPREAD, False, WEIGHT_WHEN, WEIGHT_CACHE)


@triton.jit
def _activation(A, m, step, M: tl.constexpr, BM: tl.constexpr, ASYNC: tl.constexpr,
                ACT_LAYOUT: tl.constexpr):
    if ACT_LAYOUT == 1:
        base = A + step // 2 * BM * 64
        row_width: tl.constexpr = 64
        start = step % 2 * 32
    elif ACT_LAYOUT == 2:
        base = A + step * BM * 32
        row_width: tl.constexpr = 32
        start = 0
    else:
        base = A
        row_width: tl.constexpr = 320
        start = step * 32
    if ASYNC and M == BM:
        block = tl.make_block_ptr(base, (BM, row_width), (row_width, 1),
                                  (0, start), (BM, 32), (1, 0))
        act = tl.aiu_load(block)
    else:
        k = tl.arange(0, 32)
        offsets = m[:, None] * row_width + start + k[None, :]
        act = tl.load(base + offsets, m[:, None] < M, 0, cache_modifier=".cg")
    return act


@triton.jit
def _mega_sigmoid(value):
    rounded = value.to(tl.bfloat16).to(tl.float32)
    return libdevice.fast_dividef(1.0, 1.0 + libdevice.fast_expf(-rounded))


@triton.jit
def _read_page(W, tile, step, AIU: tl.constexpr):
    base = W + tile * 128 * 320 + step * 4096
    if AIU:
        page = tl.make_block_ptr(base, (128, 32), (32, 1),
                                 (0, 0), (128, 32), (1, 0))
        return tl.trans(tl.aiu_load(page))
    else:
        n = tl.arange(0, 128)
        k = tl.arange(0, 32)
        return tl.load(base + n[None, :] * 32 + k[:, None], cache_modifier=".cg")


@triton.jit
def _unpack_four(v, BM: tl.constexpr, LAYOUT: tl.constexpr):
    if LAYOUT == 0:
        branches = tl.permute(tl.reshape(v, (BM, 2, 2, 32)), (0, 3, 1, 2))
        even, odd = tl.split(branches)
    elif LAYOUT == 1:
        even, odd = tl.split(tl.reshape(v, (BM, 2, 32, 2)))
        even = tl.permute(even, (0, 2, 1))
        odd = tl.permute(odd, (0, 2, 1))
    else:
        even, odd = tl.split(tl.reshape(v, (BM, 32, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    return a0, a1, a2, a3


@triton.jit
def _wait(counter, epoch):
    initial = tl.atomic_add(counter, 0, sem="acquire", scope="gpu").to(tl.uint32)
    _mega_wait(counter, initial, epoch)


@triton.jit
def _up(A, W, X, Y, S, epoch, tile, physical, M: tl.constexpr, BM: tl.constexpr,
        DEPTH: tl.constexpr, LAYOUT: tl.constexpr,
        EPILOGUE: tl.constexpr,
        HEAD: tl.constexpr, ACT_AIU: tl.constexpr, STATE_STRIDE: tl.constexpr,
        ACT_LAYOUT: tl.constexpr):
    m = tl.arange(0, BM)
    j = tile * 32 + tl.arange(0, 32)
    if HEAD:
        b0 = _read_page(W, tile, 0, False)
    _wait(S + 7 * STATE_STRIDE, epoch)
    if HEAD:
        acc = tl.dot(_activation(A, m, 0, M, BM, False, ACT_LAYOUT), b0)
    else:
        acc = tl.zeros((BM, 128), tl.float32)
    for step in tl.range(1 if HEAD else 0, 10, num_stages=DEPTH):
        b = _read_page(W, tile, step, False)
        acc = tl.dot(_activation(A, m, step, M, BM, ACT_AIU, ACT_LAYOUT), b, acc)
    p0, p1, p2, p3 = _unpack_four(acc, BM, LAYOUT)
    x = X + m[:, None] * 10240 + j[None, :]
    valid = m[:, None] < M
    if EPILOGUE == 0:
        s0 = _mega_sigmoid(p0)
        s1 = _mega_sigmoid(p1)
        s2 = _mega_sigmoid(p2)
        s3 = _mega_sigmoid(p3)
        x0 = tl.load(x, valid, 0, cache_modifier=".cg").to(tl.float32)
        x1 = tl.load(x + 2560, valid, 0, cache_modifier=".cg").to(tl.float32)
        x2 = tl.load(x + 5120, valid, 0, cache_modifier=".cg").to(tl.float32)
        x3 = tl.load(x + 7680, valid, 0, cache_modifier=".cg").to(tl.float32)
        out = tl.fma(s0, x0, s1 * x1)
        out = tl.fma(s2, x2, out)
        out = tl.fma(s3, x3, out)
    else:
        x0 = tl.load(x, valid, 0, cache_modifier=".cg").to(tl.float32)
        x1 = tl.load(x + 2560, valid, 0, cache_modifier=".cg").to(tl.float32)
        out = tl.fma(_mega_sigmoid(p0), x0, _mega_sigmoid(p1) * x1)
        if EPILOGUE == 2:
            tl.debug_barrier()
        x2 = tl.load(x + 5120, valid, 0, cache_modifier=".cg").to(tl.float32)
        out = tl.fma(_mega_sigmoid(p2), x2, out)
        x3 = tl.load(x + 7680, valid, 0, cache_modifier=".cg").to(tl.float32)
        out = tl.fma(_mega_sigmoid(p3), x3, out)
    tl.store(Y + m[:, None] * 2560 + j[None, :], out * 0.25, m[:, None] < M)


@triton.jit
def _up_assignment(pid, logical, SK: tl.constexpr, CTAS: tl.constexpr,
                   MAPPING: tl.constexpr, ROTATE: tl.constexpr, ORDER: tl.constexpr):
    fast: tl.constexpr = (SK - 8) * 6
    idle: tl.constexpr = CTAS - SK * 6
    if ORDER == 1:
        return tl.where(logical < fast, logical,
                        tl.where(logical >= SK * 6, fast + logical - SK * 6, logical + idle))
    if ORDER >= 2:
        lane = pid % 39
        start = lane + (39 if ORDER == 3 else 0)
        spare = lane + 78
        first_down = _logical_down(start, SK, MAPPING, ROTATE)
        spare_down = _logical_down(spare, SK, MAPPING, ROTATE)
        move = (spare >= 80) & (spare < CTAS) & (spare_down >= SK * 6)
        move = move & (first_down >= fast) & (first_down < SK * 6)
        return tl.where(move & (pid == start), -1,
                        tl.where(move & (pid == spare), start, pid))
    return pid


@triton.jit
def fused_gated_hc(B, R, G0, WN, WD, WG, WU, U, X, P, A, G, Y, S,
           M: tl.constexpr, BM: tl.constexpr, SK: tl.constexpr,
           DOWN_DEPTH: tl.constexpr, UP_DEPTH: tl.constexpr,
           CTAS: tl.constexpr, LAYOUT: tl.constexpr, EPILOGUE: tl.constexpr,
           GATE_PURE: tl.constexpr,
           HEAD: tl.constexpr, ACT_AIU: tl.constexpr, SPREAD: tl.constexpr,
           GROUP: tl.constexpr, TAIL_BATCH: tl.constexpr, TAIL_SPREAD: tl.constexpr,
           TAIL_BACK: tl.constexpr, TAIL_FIRST: tl.constexpr,
           MAPPING: tl.constexpr, X_CACHE: tl.constexpr,
           ROTATE: tl.constexpr, UP_ORDER: tl.constexpr,
           TAIL_WAVE: tl.constexpr, TAIL_ROT: tl.constexpr, BRANCH_MAJOR: tl.constexpr,
           REQUESTED: tl.constexpr, OWNERS: tl.constexpr, TURN: tl.constexpr,
           STATE_STRIDE: tl.constexpr, FUSE_WAIT: tl.constexpr,
           MAIN_WEIGHT: tl.constexpr, TAIL_WEIGHT: tl.constexpr, WEIGHT_CACHE: tl.constexpr,
           ACT_LAYOUT: tl.constexpr, TILE_ORDER: tl.constexpr,
           DOWN_AIU: tl.constexpr, W_CACHE: tl.constexpr, A_EVICT: tl.constexpr,
           W_EVICT: tl.constexpr):
    pid = tl.program_id(0)
    if TAIL_FIRST:
        _norm_tail(B, R, G0, WN, U, X, pid, M, CTAS, TAIL_BATCH, TAIL_SPREAD, TAIL_BACK,
                   TAIL_WAVE, TAIL_ROT, BRANCH_MAJOR, TAIL_WEIGHT, WEIGHT_CACHE)
    if pid < M:
        _warp_norm(B, R, G0, WN, U, X, pid, 4, SPREAD, GATE_PURE, MAIN_WEIGHT, WEIGHT_CACHE)
    if not TAIL_FIRST:
        _norm_tail(B, R, G0, WN, U, X, pid, M, CTAS, TAIL_BATCH, TAIL_SPREAD, TAIL_BACK,
                   TAIL_WAVE, TAIL_ROT, BRANCH_MAJOR, TAIL_WEIGHT, WEIGHT_CACHE)
    tl.debug_barrier()
    if CTAS == 96:
        count = tl.where(pid < 64, 6, 4)
    else:
        count = 512 // CTAS + (pid < 512 % CTAS).to(tl.int32)
    serial = tl.atomic_add(S, count, sem="acq_rel", scope="gpu").to(tl.uint32)
    epoch = (serial & 0xfffffe00) + tl.full((), 512, tl.uint32)
    logical = _logical_down(pid, SK, MAPPING, ROTATE)
    running = logical < SK * 6
    if running:
        _down(X, WD, WG, P, A, G, S, pid, logical, serial + count.to(tl.uint32), epoch,
              M, BM, SK, DOWN_DEPTH, GROUP, X_CACHE, REQUESTED, OWNERS, TURN,
              STATE_STRIDE, FUSE_WAIT, ACT_LAYOUT, DOWN_AIU, W_CACHE, A_EVICT, W_EVICT)
    up_tile = _up_assignment(pid, logical, SK, CTAS, MAPPING, ROTATE, UP_ORDER)
    if (up_tile >= 0) & (up_tile < 80):
        if TILE_ORDER:
            up_tile = (up_tile % TILE_ORDER) * (80 // TILE_ORDER) + up_tile // TILE_ORDER
        _up(A, WU, X, Y, S, epoch, up_tile, pid, M, BM, UP_DEPTH,
            LAYOUT, EPILOGUE, HEAD, ACT_AIU, STATE_STRIDE, ACT_LAYOUT)


@lru_cache(maxsize=None)
def _device_supported(device_index: int) -> bool:
    properties = torch.cuda.get_device_properties(device_index)
    return (
        (properties.major, properties.minor) == (8, 9)
        and properties.multi_processor_count == 39
    )


# Cooperative launches that share this buffer are ordered on its owning stream.
# The counters advance across graph replays and are initialized before capture.
_workspaces = {}


def _workspace(device, rows, split_k):
    bm = max(32, triton.next_power_of_2(rows))
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream, bm, split_k)
    workspace = _workspaces.get(key)
    if workspace is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        partials = torch.empty((6, split_k, bm, 64), device=device, dtype=torch.float32)
        activation = torch.empty((bm, 320), device=device, dtype=torch.bfloat16)
        counters = torch.zeros(9 * 48, dtype=torch.int32, device=device)
        counters[::48] = torch.tensor(
            [-512, *([-split_k] * 6), -512, 0], dtype=torch.int32, device=device
        )
        workspace = (partials, activation, counters)
        _workspaces[key] = workspace
    return workspace


def run_gated_hc(block, residual, old_gates, norm, down, gate, up, owner):
    rows = block.shape[0]
    if (
        block.ndim != 2
        or not block.is_cuda
        or not 1 <= rows <= 128
        or block.shape[1] != 2560
        or residual.shape != (rows, 10240)
        or old_gates.shape != (rows, 4)
        or old_gates.dtype != torch.float32
        or norm.shape != (10240,)
        or down.shape != (320, 10240)
        or gate.shape != (4, 10240)
        or up.shape != (10240, 320)
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not _device_supported(block.device.index)
        or any(
            value.dtype != torch.bfloat16
            or value.device != block.device
            or not value.is_contiguous()
            for value in (block, residual, norm, down, gate, up)
        )
        or old_gates.device != block.device
        or not old_gates.is_contiguous()
    ):
        return None

    version = None if up.is_inference() else up._version
    cached = getattr(owner, "_mega_packed_up", None)
    if cached is not None and cached[0] is up and cached[1] == version:
        packed = cached[2]
    elif torch.cuda.is_current_stream_capturing():
        _log_capture_fallback("packed up-projection weights not warmed up", rows)
        return None
    else:
        packed = up.view(2, 2, 80, 32, 10, 32).permute(2, 4, 0, 3, 1, 5).contiguous()
        owner._mega_packed_up = (up, version, packed)

    split_k = _pick_config(rows)[4]
    workspace = _workspace(block.device, rows, split_k)
    if workspace is None:
        _log_capture_fallback("workspace not warmed up on this stream", rows)
        return None

    partials, activation, counters = workspace
    updated = torch.empty_like(residual)
    normed = torch.empty_like(residual)
    gates = torch.empty_like(old_gates)
    mixed = torch.empty_like(block)
    fused_gated_hc[(117,)](
        block, residual, old_gates, norm, down, gate, packed,
        updated, normed, partials, activation, gates, mixed, counters,
        M=rows, BM=max(32, triton.next_power_of_2(rows)), SK=split_k,
        DOWN_DEPTH=3, UP_DEPTH=4,
        CTAS=117, LAYOUT=1, EPILOGUE=0, GATE_PURE=False,
        HEAD=False, ACT_AIU=True, SPREAD=2, GROUP=0, TAIL_BATCH=1, TAIL_SPREAD=8,
        TAIL_BACK=False, TAIL_FIRST=True, MAPPING=4, X_CACHE=".cg",
        ROTATE=0, UP_ORDER=0, TAIL_WAVE=3, TAIL_ROT=18, BRANCH_MAJOR=False,
        REQUESTED=16, OWNERS=0, TURN=0, STATE_STRIDE=48, FUSE_WAIT=False,
        MAIN_WEIGHT=1, TAIL_WEIGHT=1, WEIGHT_CACHE="", ACT_LAYOUT=0, TILE_ORDER=8,
        DOWN_AIU=1, W_CACHE=".cg", A_EVICT="", W_EVICT="evict_first",
        num_warps=8, num_stages=4, launch_cooperative_grid=True,
        enable_fp_fusion=False, maxnreg=160,
    )
    return updated, normed, mixed, gates
