//! Prefill-backlog aware routing policy.
//!
//! For every healthy worker `w` the policy minimises
//!
//! ```text
//! score(w) = inflight_chars(w) + weight * (prompt_chars - prefix_match_chars(w))
//! ```
//!
//! * `inflight_chars(w)` is the number of characters the requests already
//!   dispatched to `w` still have to prefill — the same unit as the second
//!   term, so a request whose prefix is fully resident costs `w` nothing. It is
//!   maintained by this policy: added in `select_worker`, removed through
//!   `on_request_complete` (wired from the router's per-request load guard).
//!   Entries older than `pending_ttl_secs` are pruned lazily so a lost
//!   completion cannot leak.
//! * `matched` comes from one radix tree per pool/model that records what the
//!   *cluster* has seen, not what one worker owns: with a shared storage tier a
//!   prefix is reachable from any instance, so the work a request needs is
//!   `prompt_chars - matched` regardless of where it lands.
//!
//! Characters (not tokens) are used on purpose: the router has no tokenizer and
//! the tree already works in characters, so every candidate is compared on the
//! same, cheap yardstick.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use dashmap::DashMap;
use rand::Rng;
use tracing::debug;

use super::{
    get_healthy_worker_indices, normalize_model_key, tree::Tree, utils::PeriodicTask,
    LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::config::PrefillSelection;
use crate::core::{Worker, WorkerType};

#[derive(Debug, Clone)]
pub struct PrefillAwareConfig {
    /// Weight of a not-yet-prefilled character relative to one queued character.
    pub prefill_weight: f64,
    /// Interval of the tree eviction task (0 disables it).
    pub eviction_interval_secs: u64,
    /// Maximum characters kept per tenant in one pool/model tree.
    ///
    /// Same limit `cache_aware` applies, and also per tenant: N workers on one
    /// pool/model may hold up to N times this. Tenants are keyed by worker url,
    /// so a worker removed from the pool leaves its tree behind until the
    /// eviction task trims it.
    pub max_tree_size: usize,
    /// Pending entries older than this are pruned defensively.
    ///
    /// Must exceed the router's request timeout: pruning an entry whose request
    /// is still running would let that request's later completion drain another
    /// request's entry.
    pub pending_ttl_secs: u64,
    /// Relative tolerance used to randomise among near-identical scores
    /// (`Weighted`) or near-identical backlogs (`Lexicographic`).
    pub tie_tolerance_ratio: f64,
    /// Ranking mode.
    pub selection: PrefillSelection,
    /// Backlogs within this many tokens of the smallest are treated as equal in
    /// `Lexicographic` mode. 0 = strict equality.
    pub backlog_epsilon_tokens: u64,
}

impl Default for PrefillAwareConfig {
    fn default() -> Self {
        Self {
            prefill_weight: 1.0,
            eviction_interval_secs: 30,
            max_tree_size: 10_000_000,
            // Keep well above the default request timeout (600 s): pruning a
            // live request under-counts that worker for the rest of the run.
            pending_ttl_secs: 3_600,
            tie_tolerance_ratio: 0.02,
            selection: PrefillSelection::default(),
            backlog_epsilon_tokens: 0,
        }
    }
}

/// Outstanding dispatches for one worker.
///
/// Every dispatch gets a unique id and its completion removes exactly that
/// entry, so out-of-order completions cannot mis-attribute characters: a short
/// request finishing first no longer pops a long request's reservation. The
/// queue also bounds the sum when a completion never arrives (a request
/// outliving the TTL): entries are pruned oldest-first and a completion whose
/// entry was already pruned is a no-op. `pending_ttl_secs` must still stay
/// above the router's request timeout so pruning does not race live requests.
#[derive(Debug, Clone, Copy)]
struct Entry {
    started: u64,
    chars: u64,
    /// True once the first response byte was seen: prefill is finished and the
    /// request now only occupies a decode slot.
    decoding: bool,
}

#[derive(Debug)]
struct Pending {
    entries: Mutex<HashMap<u64, Entry>>,
    /// Characters of *every* outstanding dispatch; released on completion.
    inflight: AtomicU64,
    /// Characters of dispatches still prefilling; released on the first byte.
    pending_prefill: AtomicU64,
    /// Dispatches that have not produced their first byte yet.
    prefilling: AtomicU64,
    /// Dispatches past their first byte but not finished.
    decoding: AtomicU64,
    next_id: AtomicU64,
}

impl Pending {
    fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            inflight: AtomicU64::new(0),
            pending_prefill: AtomicU64::new(0),
            prefilling: AtomicU64::new(0),
            decoding: AtomicU64::new(0),
            next_id: AtomicU64::new(1),
        }
    }

    /// Reserve `chars` for one dispatch and return its id.
    fn push(&self, chars: u64) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let entry = Entry {
            started: epoch_secs(),
            chars,
            decoding: false,
        };
        let mut entries = self.entries.lock().unwrap();
        entries.insert(id, entry);
        self.inflight.fetch_add(chars, Ordering::Relaxed);
        self.pending_prefill.fetch_add(chars, Ordering::Relaxed);
        self.prefilling.fetch_add(1, Ordering::Relaxed);
        id
    }

    /// The dispatch produced its first byte: prefill is done, decode begins.
    /// Unknown ids (already released) are a no-op, which also makes a repeated
    /// signal harmless.
    fn first_token(&self, id: u64) {
        let mut entries = self.entries.lock().unwrap();
        if let Some(entry) = entries.get_mut(&id) {
            if !entry.decoding {
                entry.decoding = true;
                self.pending_prefill
                    .fetch_sub(entry.chars, Ordering::Relaxed);
                self.prefilling.fetch_sub(1, Ordering::Relaxed);
                self.decoding.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Release the reservation with `id`, whichever phase it was in.
    fn complete(&self, id: u64) {
        let mut entries = self.entries.lock().unwrap();
        if let Some(entry) = entries.remove(&id) {
            self.inflight.fetch_sub(entry.chars, Ordering::Relaxed);
            if entry.decoding {
                self.decoding.fetch_sub(1, Ordering::Relaxed);
            } else {
                self.pending_prefill
                    .fetch_sub(entry.chars, Ordering::Relaxed);
                self.prefilling.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Release the oldest reservation. Only used when a path cannot supply the
    /// dispatch id; completion order then decides which slot is released.
    fn pop_front(&self) {
        let mut entries = self.entries.lock().unwrap();
        // Ids increase with dispatch order, so the smallest id is the oldest
        // reservation; `started` only has second granularity and would tie.
        let oldest = entries
            .iter()
            .min_by_key(|(id, _)| **id)
            .map(|(id, entry)| (*id, *entry));
        if let Some((id, entry)) = oldest {
            entries.remove(&id);
            self.inflight.fetch_sub(entry.chars, Ordering::Relaxed);
            if entry.decoding {
                self.decoding.fetch_sub(1, Ordering::Relaxed);
            } else {
                self.pending_prefill
                    .fetch_sub(entry.chars, Ordering::Relaxed);
                self.prefilling.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    fn prune(&self, ttl_secs: u64) {
        if ttl_secs == 0 {
            return;
        }
        let now = epoch_secs();
        let mut entries = self.entries.lock().unwrap();
        let expired: Vec<(u64, Entry)> = entries
            .iter()
            .filter(|(_, entry)| now.saturating_sub(entry.started) >= ttl_secs)
            .map(|(id, entry)| (*id, *entry))
            .collect();
        for (id, entry) in expired {
            entries.remove(&id);
            self.inflight.fetch_sub(entry.chars, Ordering::Relaxed);
            if entry.decoding {
                self.decoding.fetch_sub(1, Ordering::Relaxed);
            } else {
                self.pending_prefill
                    .fetch_sub(entry.chars, Ordering::Relaxed);
                self.prefilling.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Characters of all outstanding dispatches (current score input).
    fn load(&self) -> u64 {
        self.inflight.load(Ordering::Relaxed)
    }

    /// Characters still to be prefilled (released on the first byte).
    fn pending_prefill_chars(&self) -> u64 {
        self.pending_prefill.load(Ordering::Relaxed)
    }

    /// (prefilling, decoding) dispatch counts.
    fn counts(&self) -> (u64, u64) {
        (
            self.prefilling.load(Ordering::Relaxed),
            self.decoding.load(Ordering::Relaxed),
        )
    }
}

/// Characters per token used to turn token thresholds into character sums; the
/// router has no tokenizer.
const CHARS_PER_TOKEN: u64 = 3;

/// Single tenant name for the cluster-wide view of each tree. Prefixes are
/// stored once, not per worker: with a shared storage tier, "seen by anyone"
/// is what decides whether a request has to compute its prefix.
const CLUSTER_TENANT: &str = "cluster";

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn pool_tag(worker_type: &WorkerType) -> &'static str {
    match worker_type {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    }
}

fn make_tree_key(pool: &str, model: &str) -> String {
    format!("{}::{}", pool, model)
}

fn tree_key_for_worker(worker: &dyn Worker) -> String {
    make_tree_key(
        pool_tag(worker.worker_type()),
        normalize_model_key(worker.model_id()),
    )
}

/// Prefill-backlog aware policy: queues work where the (prefix match +
/// outstanding characters) sum is smallest.
#[derive(Debug)]
pub struct PrefillAwarePolicy {
    config: PrefillAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    pending: DashMap<String, Arc<Pending>>,
    _eviction_task: Option<PeriodicTask>,
}

impl PrefillAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(PrefillAwareConfig::default())
    }

    pub fn with_config(config: PrefillAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());
        let eviction_task = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;
            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "prefill_aware_eviction",
                move || {
                    for tree_ref in trees_clone.iter() {
                        tree_ref.value().evict_tenant_by_size(max_tree_size);
                        debug!("prefill_aware eviction completed for {}", tree_ref.key());
                    }
                },
            ))
        } else {
            None
        };
        Self {
            config,
            trees,
            pending: DashMap::new(),
            _eviction_task: eviction_task,
        }
    }

    fn pending_for(&self, url: &str) -> Arc<Pending> {
        self.pending
            .entry(url.to_string())
            .or_insert_with(|| Arc::new(Pending::new()))
            .clone()
    }

    /// Outstanding estimated characters for one worker (tests and diagnostics).
    #[cfg(test)]
    pub(crate) fn inflight_chars(&self, url: &str) -> u64 {
        self.pending.get(url).map(|p| p.load()).unwrap_or(0)
    }

    /// Characters still to be prefilled on one worker (tests and diagnostics).
    #[cfg(test)]
    pub(crate) fn pending_prefill_chars(&self, url: &str) -> u64 {
        self.pending
            .get(url)
            .map(|p| p.pending_prefill_chars())
            .unwrap_or(0)
    }

    /// (prefilling, decoding) dispatch counts for one worker.
    #[cfg(test)]
    pub(crate) fn phase_counts(&self, url: &str) -> (u64, u64) {
        self.pending.get(url).map(|p| p.counts()).unwrap_or((0, 0))
    }
}

impl Default for PrefillAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoadBalancingPolicy for PrefillAwarePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.select_worker_with_token(workers, info)
            .await
            .map(|(idx, _)| idx)
    }

    async fn select_worker_with_token(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<(usize, Option<u64>)> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }

        let tree_key = tree_key_for_worker(workers[healthy[0]].as_ref());
        let tree = self
            .trees
            .entry(tree_key.clone())
            .or_insert_with(|| Arc::new(Tree::new()))
            .clone();

        let text = info.request_text.unwrap_or("");
        let prompt_chars: u64 = text.chars().count() as u64;

        // One cluster-wide lookup: how much of this prompt nobody has seen yet.
        // With a shared storage tier the prefix is reachable from any instance,
        // so the remaining prefill work does not depend on the placement.
        let matched = tree.prefix_match_chars_for_tenant(text, CLUSTER_TENANT, false) as u64;
        let new_chars = prompt_chars.saturating_sub(matched);

        // Per-candidate load: the ledger keeps prefill and decode apart.
        struct Cand {
            idx: usize,
            pending_prefill: u64,
            decoding: u64,
            outstanding: u64,
        }
        let mut cands: Vec<Cand> = Vec::with_capacity(healthy.len());
        for &idx in &healthy {
            let w = workers[idx].as_ref();
            let pend = self.pending_for(w.url());
            pend.prune(self.config.pending_ttl_secs);
            let (_, decoding) = pend.counts();
            cands.push(Cand {
                idx,
                pending_prefill: pend.pending_prefill_chars(),
                decoding,
                outstanding: pend.load(),
            });
        }

        let mut rng = rand::rng();
        let chosen_entry = match self.config.selection {
            PrefillSelection::Weighted => {
                let scored: Vec<(usize, f64)> = cands
                    .iter()
                    .map(|c| {
                        (
                            c.idx,
                            c.outstanding as f64 + self.config.prefill_weight * new_chars as f64,
                        )
                    })
                    .collect();
                let best = scored.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
                let tol = (self.config.tie_tolerance_ratio * prompt_chars as f64).max(1.0);
                let pool: Vec<usize> = scored
                    .iter()
                    .filter(|(_, s)| *s <= best + tol)
                    .map(|(i, _)| *i)
                    .collect();
                let idx = if pool.is_empty() {
                    scored[0].0
                } else {
                    pool[rng.random_range(0..pool.len())]
                };
                cands.iter().find(|c| c.idx == idx).unwrap()
            }
            PrefillSelection::Lexicographic => {
                let mut pool: Vec<&Cand> = cands.iter().collect();
                // Level 1: least pending prefill (within the epsilon deadband).
                if new_chars > 0 {
                    let min_pending = pool.iter().map(|c| c.pending_prefill).min().unwrap();
                    let eps = self
                        .config
                        .backlog_epsilon_tokens
                        .saturating_mul(CHARS_PER_TOKEN);
                    pool.retain(|c| c.pending_prefill <= min_pending.saturating_add(eps));
                }
                // Level 2: fewest decode dispatches.
                let min_decoding = pool.iter().map(|c| c.decoding).min().unwrap();
                pool.retain(|c| c.decoding <= min_decoding);
                pool[rng.random_range(0..pool.len())]
            }
        };

        let chosen = chosen_entry.idx;
        let pending_chars = new_chars;
        let chosen_url = workers[chosen].url().to_string();
        // Account only what this worker still has to prefill. The entry is
        // pushed even when it is zero: completions pop one entry each, so a
        // skipped push would let a later completion drain another request's
        // entry.
        let dispatch_id = self.pending_for(&chosen_url).push(pending_chars);
        tree.insert(text, CLUSTER_TENANT);
        workers[chosen].increment_processed();

        Some((chosen, Some(dispatch_id)))
    }

    fn on_request_complete(&self, worker_url: &str, _success: bool) {
        if let Some(p) = self.pending.get(worker_url) {
            p.pop_front();
        }
    }

    fn on_first_token(&self, worker_url: &str, dispatch_id: Option<u64>) {
        if let (Some(id), Some(p)) = (dispatch_id, self.pending.get(worker_url)) {
            p.first_token(id);
        }
    }

    fn on_dispatch_complete(&self, worker_url: &str, dispatch_id: Option<u64>, success: bool) {
        match (dispatch_id, self.pending.get(worker_url)) {
            (Some(id), Some(p)) => p.complete(id),
            // No token (router that cannot carry one) or the worker is gone:
            // fall back to releasing the oldest reservation.
            _ => self.on_request_complete(worker_url, success),
        }
    }

    fn name(&self) -> &'static str {
        "prefill_aware"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .build(),
        )
    }

    fn info(text: &str) -> SelectWorkerInfo<'_> {
        SelectWorkerInfo {
            request_text: Some(text),
            ..Default::default()
        }
    }

    /// A short request must avoid the worker still carrying a long prefill.
    ///
    /// The margin is deliberate: with 100 vs 10 characters the two scores are
    /// `100 + 0` and `0 + 10`, far outside the 2 % tie tolerance, so the
    /// outcome cannot be decided by the random tie-break.
    #[tokio::test]
    async fn routes_away_from_queued_worker() {
        let policy = PrefillAwarePolicy::new();
        let workers = vec![worker("http://w1:8000"), worker("http://w2:8000")];
        let long = "x".repeat(100);
        let first = policy.select_worker(&workers, &info(&long)).await.unwrap();
        let second = policy
            .select_worker(&workers, &info("abcdefghij"))
            .await
            .unwrap();
        assert_ne!(
            first, second,
            "short request must avoid the worker carrying a long prefill"
        );

        policy.on_request_complete(workers[first].url(), true);
        policy.on_request_complete(workers[second].url(), true);
        assert_eq!(policy.inflight_chars(workers[first].url()), 0);
        assert_eq!(policy.inflight_chars(workers[second].url()), 0);
    }

    /// Only the part that still has to be prefilled is accounted: a request
    /// whose prefix is already in the tree costs its worker nothing.
    #[tokio::test]
    async fn ledger_records_only_unmatched_chars() {
        let policy = PrefillAwarePolicy::new();
        let w1 = worker("http://w1:8000");
        let text = "x".repeat(100);

        // Cold tree: the whole prompt has to be prefilled.
        policy
            .select_worker(std::slice::from_ref(&w1), &info(&text))
            .await
            .unwrap();
        assert_eq!(policy.inflight_chars(w1.url()), 100);
        policy.on_request_complete(w1.url(), true);
        assert_eq!(policy.inflight_chars(w1.url()), 0);

        // Same prompt again: the tree already owns the prefix for w1.
        policy
            .select_worker(std::slice::from_ref(&w1), &info(&text))
            .await
            .unwrap();
        assert_eq!(policy.inflight_chars(w1.url()), 0);
    }

    /// Out-of-order completions still drain the ledger: only the sum is scored.
    #[test]
    fn out_of_order_completion_drains_to_zero() {
        let p = Pending::new();
        p.push(100);
        p.push(7);
        assert_eq!(p.load(), 107);
        p.pop_front();
        assert_eq!(p.load(), 7);
        p.pop_front();
        assert_eq!(p.load(), 0);
    }

    /// Completions release their own reservation: a short request finishing
    /// first must not release the long request's characters.
    #[test]
    fn completion_by_id_is_order_independent() {
        let p = Pending::new();
        let long = p.push(100);
        let short = p.push(7);
        assert_eq!(p.load(), 107);

        p.complete(short);
        assert_eq!(p.load(), 100, "the long reservation must survive");

        p.complete(long);
        assert_eq!(p.load(), 0);
    }

    /// A completion whose entry was already pruned is a no-op, so it cannot
    /// drain another request's reservation.
    #[tokio::test]
    async fn completion_after_prune_is_noop() {
        let p = Pending::new();
        let stale = p.push(42);
        tokio::time::sleep(std::time::Duration::from_millis(1_200)).await;
        let fresh = p.push(5);
        p.prune(1);
        assert_eq!(p.load(), 5);

        p.complete(stale);
        assert_eq!(
            p.load(),
            5,
            "a pruned dispatch must not release the fresh one"
        );

        p.complete(fresh);
        assert_eq!(p.load(), 0);
    }

    /// The first byte ends the prefill phase: the characters leave the pending
    /// prefill sum but stay counted as an in-flight dispatch until completion.
    #[test]
    fn first_token_moves_the_dispatch_to_decode() {
        let p = Pending::new();
        let id = p.push(120);
        assert_eq!(p.pending_prefill_chars(), 120);
        assert_eq!(p.counts(), (1, 0));

        p.first_token(id);
        assert_eq!(p.pending_prefill_chars(), 0);
        assert_eq!(p.load(), 120, "the dispatch is still outstanding");
        assert_eq!(p.counts(), (0, 1));

        p.first_token(id); // repeated signal must not double count
        assert_eq!(p.counts(), (0, 1));

        p.complete(id);
        assert_eq!(p.load(), 0);
        assert_eq!(p.counts(), (0, 0));
    }

    /// A dispatch that finishes without a first byte releases its prefill share.
    #[test]
    fn completion_without_first_token_releases_prefill() {
        let p = Pending::new();
        let id = p.push(80);
        p.complete(id);
        assert_eq!(p.pending_prefill_chars(), 0);
        assert_eq!(p.counts(), (0, 0));
        assert_eq!(p.load(), 0);
    }

    /// The policy hooks forward to the entry issued by selection.
    #[tokio::test]
    async fn policy_splits_prefill_and_decode_on_first_token() {
        let policy = PrefillAwarePolicy::new();
        let w = worker("http://w1:8000");
        let (_, token) = policy
            .select_worker_with_token(std::slice::from_ref(&w), &info(&"x".repeat(90)))
            .await
            .unwrap();
        let url = w.url().to_string();
        assert_eq!(policy.pending_prefill_chars(&url), 90);
        assert_eq!(policy.phase_counts(&url), (1, 0));

        policy.on_first_token(&url, token);
        assert_eq!(policy.pending_prefill_chars(&url), 0);
        assert_eq!(policy.phase_counts(&url), (0, 1));
        assert_eq!(policy.inflight_chars(&url), 90);

        policy.on_dispatch_complete(&url, token, true);
        assert_eq!(policy.phase_counts(&url), (0, 0));
        assert_eq!(policy.inflight_chars(&url), 0);
    }

    /// Completions without a matching entry must neither panic nor wrap the
    /// counter below zero.
    #[test]
    fn completion_without_pending_entry_is_noop() {
        let policy = PrefillAwarePolicy::new();
        policy.on_request_complete("http://nope:8000", true);

        let p = Pending::new();
        p.pop_front();
        p.pop_front();
        assert_eq!(p.load(), 0);
    }

    /// Only entries older than the TTL are pruned; `ttl == 0` disables it.
    #[tokio::test]
    async fn prune_drops_only_stale_entries() {
        let p = Pending::new();
        p.push(42);
        tokio::time::sleep(std::time::Duration::from_millis(1_200)).await;
        p.push(5);
        p.prune(1);
        assert_eq!(p.load(), 5, "only the stale entry may be pruned");
        p.prune(0);
        assert_eq!(p.load(), 5, "ttl 0 disables pruning");
    }

    /// Lexicographic level 1: a worker already owing prefill loses to an idle
    /// one, regardless of the prompt length.
    #[tokio::test]
    async fn lexicographic_prefers_least_pending_prefill() {
        let policy = PrefillAwarePolicy::new();
        let w1 = worker("http://w1:8000");
        let w2 = worker("http://w2:8000");
        let url1 = w1.url().to_string();

        policy
            .select_worker(std::slice::from_ref(&w1), &info(&"a".repeat(300)))
            .await
            .unwrap();
        assert_eq!(policy.pending_prefill_chars(&url1), 300);

        // A different prompt matches nothing, so both workers owe it in full.
        let (idx, _) = policy
            .select_worker_with_token(&[w1.clone(), w2.clone()], &info(&"b".repeat(300)))
            .await
            .unwrap();
        assert_eq!(idx, 1, "the worker with no prefill backlog must win");
    }

    /// Lexicographic level 2: with equal prefill backlogs the tie is broken by
    /// the number of decode dispatches.
    #[tokio::test]
    async fn lexicographic_breaks_prefill_ties_on_decoding() {
        let policy = PrefillAwarePolicy::new();
        let w1 = worker("http://w1:8000");
        let w2 = worker("http://w2:8000");
        let url1 = w1.url().to_string();

        let (_, token) = policy
            .select_worker_with_token(std::slice::from_ref(&w1), &info(&"a".repeat(50)))
            .await
            .unwrap();
        policy.on_first_token(&url1, token);
        assert_eq!(policy.phase_counts(&url1), (0, 1));

        let (idx, _) = policy
            .select_worker_with_token(&[w1.clone(), w2.clone()], &info(&"b".repeat(50)))
            .await
            .unwrap();
        assert_eq!(idx, 1, "both owe prefill; the one without decode wins");
    }

    /// A prompt that is fully cached needs no prefill, so the backlog tier is
    /// skipped and selection goes straight to the decode tier.
    #[tokio::test]
    async fn lexicographic_skips_backlog_when_nothing_to_prefill() {
        let policy = PrefillAwarePolicy::new();
        let w1 = worker("http://w1:8000");
        let w2 = worker("http://w2:8000");
        let url1 = w1.url().to_string();
        let url2 = w2.url().to_string();
        let prompt = "c".repeat(200);

        // w1 owes a long prefill and the prompt enters the cluster tree.
        policy
            .select_worker(std::slice::from_ref(&w1), &info(&prompt))
            .await
            .unwrap();
        assert_eq!(policy.pending_prefill_chars(&url1), 200);

        // w2 carries two decode dispatches.
        for _ in 0..2 {
            let (_, token) = policy
                .select_worker_with_token(std::slice::from_ref(&w2), &info(&"d".repeat(20)))
                .await
                .unwrap();
            policy.on_first_token(&url2, token);
        }
        assert_eq!(policy.phase_counts(&url2), (0, 2));

        let (idx, _) = policy
            .select_worker_with_token(&[w1.clone(), w2.clone()], &info(&prompt))
            .await
            .unwrap();
        assert_eq!(idx, 0, "nothing to prefill: the decode tier decides");
    }

    /// The epsilon deadband keeps near-equal backlogs in the same tier so the
    /// decode tier can still break the tie.
    #[tokio::test]
    async fn lexicographic_backlog_epsilon_widens_the_first_tier() {
        let w1 = worker("http://w1:8000");
        let w2 = worker("http://w2:8000");
        let url1 = w1.url().to_string();

        // w1: one decode dispatch, no prefill; w2: 300 chars of prefill.
        let policy = PrefillAwarePolicy::with_config(PrefillAwareConfig {
            backlog_epsilon_tokens: 100, // 300 chars
            ..Default::default()
        });
        let (_, token) = policy
            .select_worker_with_token(std::slice::from_ref(&w1), &info(&"a".repeat(200)))
            .await
            .unwrap();
        policy.on_first_token(&url1, token);
        policy
            .select_worker_with_token(std::slice::from_ref(&w2), &info(&"b".repeat(300)))
            .await
            .unwrap();

        // Strictly, w1 has the least backlog; within the 300-char deadband
        // both stay in tier 1 and the decode count picks w2.
        let (idx, _) = policy
            .select_worker_with_token(&[w1.clone(), w2.clone()], &info(&"e".repeat(50)))
            .await
            .unwrap();
        assert_eq!(idx, 1, "within the deadband, decode count decides");

        // With the deadband removed the strict backlog wins again.
        let policy = PrefillAwarePolicy::new();
        let (_, token) = policy
            .select_worker_with_token(std::slice::from_ref(&w1), &info(&"a".repeat(200)))
            .await
            .unwrap();
        policy.on_first_token(&url1, token);
        policy
            .select_worker_with_token(std::slice::from_ref(&w2), &info(&"b".repeat(300)))
            .await
            .unwrap();
        let (idx, _) = policy
            .select_worker_with_token(&[w1.clone(), w2.clone()], &info(&"e".repeat(50)))
            .await
            .unwrap();
        assert_eq!(idx, 0, "no deadband: the lighter backlog wins");
    }
}
