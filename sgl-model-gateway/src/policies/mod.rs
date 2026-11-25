//! Load balancing policies for SGLang router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use std::{fmt::Debug, sync::Arc};

use async_trait::async_trait;
use smg_mesh::OptionalMeshSyncManager;

use crate::core::{HashRing, Worker};

mod bucket;
mod cache_aware;
mod consistent_hashing;
mod factory;
mod manual;
mod power_of_two;
mod prefill_aware;
mod prefix_hash;
mod random;
mod registry;
mod round_robin;
pub mod tree;
pub(crate) mod utils;
pub use bucket::BucketPolicy;
pub use cache_aware::CacheAwarePolicy;
pub use consistent_hashing::ConsistentHashingPolicy;
pub use factory::PolicyFactory;
pub use manual::{ManualConfig, ManualPolicy};
pub use power_of_two::PowerOfTwoPolicy;
pub use prefill_aware::{PrefillAwareConfig, PrefillAwarePolicy};
pub use prefix_hash::{PrefixHashConfig, PrefixHashPolicy};

/// Build the load-guard hook that closes a `prefill_aware` dispatch.
///
/// `prefill_aware` keeps its own outstanding-character ledger, which is only
/// decremented through [`LoadBalancingPolicy::on_request_complete`]. Every
/// router that hands a request to a worker must therefore hand this hook to
/// `WorkerLoadGuard::with_on_drop` (or `LoadGuards`); skipping a path leaks
/// its dispatches until the ledger's TTL prunes them.
///
/// Returns `None` for every other policy: their completion hook is either a
/// no-op or already covered by the worker's own load counter.
/// Build the callback that tells the policy its dispatch started producing
/// output (first SSE chunk for streaming responses).
pub fn first_token_hook(
    policy: &Arc<dyn LoadBalancingPolicy>,
    worker: &dyn Worker,
    dispatch_id: Option<u64>,
) -> Option<Box<dyn FnOnce() + Send + Sync>> {
    if policy.name() != "prefill_aware" {
        return None;
    }
    let policy = Arc::clone(policy);
    let worker_url = worker.url().to_string();
    Some(Box::new(move || {
        policy.on_first_token(&worker_url, dispatch_id)
    }))
}

pub fn completion_hook(
    policy: &Arc<dyn LoadBalancingPolicy>,
    worker: &dyn Worker,
    dispatch_id: Option<u64>,
) -> Option<Box<dyn FnOnce(bool) + Send + Sync>> {
    if policy.name() != "prefill_aware" {
        return None;
    }
    let policy = Arc::clone(policy);
    let worker_url = worker.url().to_string();
    Some(Box::new(move |success| {
        policy.on_dispatch_complete(&worker_url, dispatch_id, success)
    }))
}
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use round_robin::RoundRobinPolicy;
pub use tree::PrefixMatchResult;

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
#[async_trait]
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// Select a single worker from the available workers
    ///
    /// This is used for regular routing mode where requests go to a single worker.
    /// Now uses Arc<dyn Worker> for better performance and to avoid unnecessary cloning.
    ///
    /// # Arguments
    /// * `workers` - Available workers to select from
    /// * `info` - Additional information for routing decisions
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize>;

    /// Select a worker and, when the policy tracks per-dispatch state, the
    /// token that must be handed back on completion.
    ///
    /// Default: no token, i.e. [`Self::select_worker`] only.
    async fn select_worker_with_token(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<(usize, Option<u64>)> {
        Some((self.select_worker(workers, info).await?, None))
    }

    /// Update policy state after request completion
    ///
    /// This is called when a request completes (successfully or not) to allow
    /// policies to update their internal state.
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Default: no-op for stateless policies
    }

    /// The dispatch produced its first response byte, i.e. its prefill is done.
    ///
    /// Only policies that split prefill from decode work need this; the default
    /// is a no-op.
    fn on_first_token(&self, _worker_url: &str, _dispatch_id: Option<u64>) {}

    /// Completion signal for a dispatch that may carry a policy-issued token.
    ///
    /// Default: delegate to [`Self::on_request_complete`], which is what a
    /// policy that cannot identify the exact dispatch has to do.
    fn on_dispatch_complete(&self, worker_url: &str, _dispatch_id: Option<u64>, success: bool) {
        self.on_request_complete(worker_url, success);
    }

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    /// Check if this policy needs request text for routing decisions
    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

    /// Update worker load information
    ///
    /// This is called periodically with current load information for load-aware policies.
    fn update_loads(&self, _loads: &std::collections::HashMap<String, isize>) {
        // Default: no-op for policies that don't use load information
    }

    /// Set mesh sync manager
    fn set_mesh_sync(&mut self, _mesh_sync: OptionalMeshSyncManager) {
        // Default: no-op for policies that don't use mesh sync
    }

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub bucket_adjust_interval_secs: usize,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 5,
        }
    }
}

/// Helper function to filter healthy workers and return their indices
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy() && w.circuit_breaker().can_execute())
        .map(|(idx, _)| idx)
        .collect()
}

/// Helper function to normalize model_id to a key for policy lookups.
///
/// Returns UNKNOWN_MODEL_ID for empty model_ids to ensure consistent behavior
/// across single-model and multi-model deployments.
#[inline]
pub(crate) fn normalize_model_key(model_id: &str) -> &str {
    if model_id.is_empty() {
        crate::core::UNKNOWN_MODEL_ID
    } else {
        model_id
    }
}

/// Information passed to policy for worker selection
#[derive(Debug, Clone, Default)]
pub struct SelectWorkerInfo<'a> {
    /// Request text for cache-aware routing
    pub request_text: Option<&'a str>,
    /// Tokenized request for prefix-hash routing
    /// Used by PrefixHashPolicy for token-based prefix hashing
    pub tokens: Option<&'a [u32]>,
    /// HTTP headers for header-based routing policies
    /// Policies can extract routing information from headers like:
    /// - X-SMG-Target-Worker: Direct routing to a specific worker by index
    /// - X-SMG-Routing-Key: Consistent hash routing for session affinity
    pub headers: Option<&'a http::HeaderMap>,
    /// Pre-computed hash ring for O(log n) consistent hashing
    /// Built and cached by WorkerRegistry, passed through to avoid per-request rebuilds
    pub hash_ring: Option<Arc<HashRing>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key2")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_healthy(false);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }
}
