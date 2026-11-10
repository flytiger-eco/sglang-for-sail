"""397B dominant-only image pruning with original-grid index metadata."""

import torch


def prune_visual_tokens_dominant_only(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    pruning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep top-k visual tokens ranked by attention importance scores.

    Args:
        embeddings: [N, hidden_size] post-merger embeddings for one image.
        scores: [N] importance scores per token.
        pruning_rate: fraction of tokens to prune, in [0, 1).
                      0.0 means no pruning; 0.6 means prune 60% of tokens.

    Returns:
        pruned_embeddings: [K, hidden_size].
        keep_indices: [K] kept token indices in original order.
    """
    if not 0 <= pruning_rate < 1:
        raise ValueError("image_pruning_rate must be in [0, 1)")
    if scores.ndim != 1 or scores.numel() != embeddings.shape[0]:
        raise ValueError("One attention score is required per image embedding row")
    N = embeddings.shape[0]
    if N == 0:
        return embeddings, torch.empty(0, dtype=torch.long, device=embeddings.device)
    if pruning_rate <= 0.0:
        return embeddings, torch.arange(N, device=embeddings.device)

    keep_num = max(1, int(N * (1.0 - pruning_rate)))
    _, topk_indices = torch.topk(scores, keep_num, sorted=False)
    keep_indices = topk_indices.sort().values
    return embeddings[keep_indices], keep_indices



def prune_image_items(embeddings, scores, items, merge_size, rate):
    """Prune each original image independently and attach its sorted keep indices.

    Grid metadata is intentionally unchanged. The EPD receiver consumes the
    authoritative indices, rather than estimating counts from a rewritten grid.
    All DeepStack channels are selected with the same embedding row index.
    """
    counts_by_item = []
    for item in items:
        grids = item.image_grid_thw.reshape(-1, 3).tolist()
        counts = []
        for t, h, w in grids:
            if min(t, h, w) <= 0 or h % merge_size or w % merge_size:
                raise ValueError("Invalid image grid for spatial merge")
            counts.append(int(t * h * w) // (merge_size * merge_size))
        counts_by_item.append(counts)
    if sum(sum(c) for c in counts_by_item) != embeddings.shape[0]:
        raise ValueError("Original image grid count does not match embedding rows")
    if scores.ndim != 1 or scores.numel() != embeddings.shape[0]:
        raise ValueError("Image attention score count does not match embedding rows")
    chunks, indices_by_item = [], []
    offset = 0
    for counts in counts_by_item:
        indices = []
        for count in counts:
            pruned, keep = prune_visual_tokens_dominant_only(
                embeddings[offset:offset + count], scores[offset:offset + count], rate
            )
            chunks.append(pruned)
            indices.append(keep)
            offset += count
        indices_by_item.append(indices)
    # One GPU-to-CPU synchronization for all images, preserving part order.
    flat_indices = [i for group in indices_by_item for i in group]
    flat = torch.cat(flat_indices).cpu().tolist() if flat_indices else []
    offset = 0
    for item, indices in zip(items, indices_by_item):
        report = []
        for keep in indices:
            count = keep.numel()
            report.append(flat[offset:offset + count])
            offset += count
        item.set("image_pruning_indices", report)
    return torch.cat(chunks, dim=0) if chunks else embeddings


def should_extract_image_pruning_score(rate, language_only, use_dp, tp_size, cuda_graph):
    """P/D keep the pruning policy but never execute their visual tower."""
    if rate <= 0 or language_only:
        return False
    if use_dp or tp_size != 1 or cuda_graph:
        raise ValueError("Image pruning requires eager vision execution with TP=1 and DP disabled")
    return True
