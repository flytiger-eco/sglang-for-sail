//! Request context types for gRPC router pipeline
//!
//! This module provides the core context types that flow through the router pipeline,
//! eliminating deep parameter passing chains and providing a single source of truth
//! for request state.

use std::sync::Arc;

use axum::http::HeaderMap;

use super::{
    client::GrpcClient,
    proto_wrapper::{ProtoEmbedComplete, ProtoRequest, ProtoStream},
};
use crate::{
    core::{Worker, WorkerLoadGuard},
    policies::LoadBalancingPolicy,
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionResponse},
        classify::{ClassifyRequest, ClassifyResponse},
        embedding::{EmbeddingRequest, EmbeddingResponse},
        generate::{GenerateRequest, GenerateResponse},
        responses::ResponsesRequest,
    },
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    tokenizer::{stop::StopSequenceDecoder, traits::Tokenizer, TokenizerRegistry},
    tool_parser::ParserFactory as ToolParserFactory,
};

/// Main request processing context
///
/// This is the single source of truth for all request state as it flows
/// through the pipeline stages. Uses Rust's type system to enforce proper
/// stage ordering at compile time.
pub(crate) struct RequestContext {
    pub input: RequestInput,
    pub components: Arc<SharedComponents>,
    pub state: ProcessingState,
}

/// Immutable request input
pub(crate) struct RequestInput {
    pub request_type: RequestType,
    pub headers: Option<HeaderMap>,
    pub model_id: Option<String>,
}

/// Request type variants
/// Using Arc instead of Box to enable cheap cloning for background tasks
pub(crate) enum RequestType {
    Chat(Arc<ChatCompletionRequest>),
    Generate(Arc<GenerateRequest>),
    Responses(Arc<ResponsesRequest>),
    Embedding(Arc<EmbeddingRequest>),
    Classify(Arc<ClassifyRequest>),
}

/// Shared components (injected once at creation)
pub(crate) struct SharedComponents {
    pub tokenizer_registry: Arc<TokenizerRegistry>,
    #[allow(dead_code)]
    pub tool_parser_factory: ToolParserFactory,
    #[allow(dead_code)]
    pub reasoning_parser_factory: ReasoningParserFactory,
}

/// Mutable processing state (evolves through pipeline stages)
#[derive(Default)]
pub(crate) struct ProcessingState {
    // Stage 1: Preparation outputs
    pub preparation: Option<PreparationOutput>,

    /// Resolved tokenizer (set once in preparation, reused in response processing)
    /// This avoids redundant registry lookups across pipeline stages.
    pub tokenizer: Option<Arc<dyn Tokenizer>>,

    // Stage 2: Worker selection outputs
    pub workers: Option<WorkerSelection>,

    /// Policy instance that produced `workers`. Kept so the execution stage can
    /// close the dispatch: `prefill_aware` maintains a per-worker ledger that
    /// only its completion hook decrements.
    pub selected_policy: Option<Arc<dyn LoadBalancingPolicy>>,

    // Stage 3: Client acquisition outputs
    pub clients: Option<ClientSelection>,

    // Stage 4: Request building outputs
    pub proto_request: Option<ProtoRequest>,

    // Stage 5: Dispatch metadata
    pub dispatch: Option<DispatchMetadata>,

    // Load guard for worker load tracking (created at execution stage)
    pub load_guards: Option<LoadGuards>,

    // Stage 6: Response processing state
    pub response: ResponseState,
}

/// Output from preparation stage (Step 1)
pub(crate) struct PreparationOutput {
    /// Original text (for chat) or resolved text (for generate)
    pub original_text: Option<String>,

    /// Tokenized input
    pub token_ids: Vec<u32>,

    /// Processed messages (chat only)
    pub processed_messages: Option<super::ProcessedMessages>,

    /// Tool call constraints (if applicable)
    pub tool_constraints: Option<(String, String)>,

    /// Filtered request (if tools were filtered)
    pub filtered_request: Option<ChatCompletionRequest>,

    // Harmony-specific fields
    /// Whether this is a Harmony request (default: false)
    pub harmony_mode: bool,

    /// Selection text for worker routing (Harmony only)
    pub selection_text: Option<String>,

    /// Harmony messages for history tracking (Harmony only)
    #[allow(dead_code)]
    pub harmony_messages: Option<Vec<super::harmony::HarmonyMessage>>,

    /// Stop token IDs for Harmony models
    pub harmony_stop_ids: Option<Vec<u32>>,
}

/// Worker selection (Step 2)
pub(crate) enum WorkerSelection {
    Single {
        worker: Arc<dyn Worker>,
    },
    Dual {
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    },
}

/// Client selection (Step 3)
pub(crate) enum ClientSelection {
    Single {
        client: GrpcClient,
    },
    Dual {
        prefill: GrpcClient,
        decode: GrpcClient,
    },
}

/// Dispatch metadata (Step 5)
#[derive(Clone)]
pub(crate) struct DispatchMetadata {
    pub request_id: String,
    pub model: String,
    pub created: u64,
    pub weight_version: Option<String>,
}

/// Load guards for worker load tracking
/// Automatically decrements load when dropped
pub(crate) enum LoadGuards {
    Single {
        _guard: WorkerLoadGuard,
    },
    Dual {
        _prefill: WorkerLoadGuard,
        _decode: WorkerLoadGuard,
    },
}

#[cfg(test)]
impl LoadGuards {
    pub fn new(
        selection: &WorkerSelection,
        headers: Option<&HeaderMap>,
        policy: Option<&Arc<dyn LoadBalancingPolicy>>,
        tokens: (Option<u64>, Option<u64>),
    ) -> Self {
        let guard = |worker: &Arc<dyn Worker>, token| match policy {
            Some(policy) => WorkerLoadGuard::with_policy(worker.clone(), headers, policy, token),
            None => WorkerLoadGuard::new(worker.clone(), headers),
        };
        match selection {
            WorkerSelection::Single { worker } => Self::Single {
                _guard: guard(worker, tokens.0),
            },
            WorkerSelection::Dual { prefill, decode } => Self::Dual {
                _prefill: guard(prefill, tokens.0),
                _decode: guard(decode, tokens.1),
            },
        }
    }
}

/// Response processing state (Step 6)
#[derive(Default)]
pub(crate) struct ResponseState {
    /// Stop sequence decoder
    pub stop_decoder: Option<StopSequenceDecoder>,

    /// Execution result (streams from workers)
    pub execution_result: Option<ExecutionResult>,

    /// Final processed response
    pub final_response: Option<FinalResponse>,

    /// Responses API iteration result (Harmony only, for tool loop orchestration)
    pub responses_iteration_result: Option<super::harmony::ResponsesIterationResult>,
}

impl RequestContext {
    /// Create context for chat completion request
    pub fn for_chat(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Chat(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for generate request
    pub fn for_generate(
        request: Arc<GenerateRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Generate(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for Responses API request
    pub fn for_responses(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Responses(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for embedding request
    pub fn for_embedding(
        request: Arc<EmbeddingRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Embedding(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for classify request
    pub fn for_classify(
        request: Arc<ClassifyRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Classify(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Get chat request (panics if not chat)
    pub fn chat_request(&self) -> &ChatCompletionRequest {
        match &self.input.request_type {
            RequestType::Chat(req) => req.as_ref(),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get Arc clone of chat request (panics if not chat)
    pub fn chat_request_arc(&self) -> Arc<ChatCompletionRequest> {
        match &self.input.request_type {
            RequestType::Chat(req) => Arc::clone(req),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get generate request (panics if not generate)
    pub fn generate_request(&self) -> &GenerateRequest {
        match &self.input.request_type {
            RequestType::Generate(req) => req.as_ref(),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get Arc clone of generate request (panics if not generate)
    pub fn generate_request_arc(&self) -> Arc<GenerateRequest> {
        match &self.input.request_type {
            RequestType::Generate(req) => Arc::clone(req),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get Arc clone of responses request (panics if not responses)
    pub fn responses_request_arc(&self) -> Arc<ResponsesRequest> {
        match &self.input.request_type {
            RequestType::Responses(req) => Arc::clone(req),
            _ => panic!("Expected responses request"),
        }
    }

    /// Check if request is streaming
    pub fn is_streaming(&self) -> bool {
        match &self.input.request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Generate(req) => req.stream,
            RequestType::Responses(req) => req.stream.unwrap_or(false),
            RequestType::Embedding(_) => false, // Embeddings are never streaming
            RequestType::Classify(_) => false,  // Classification is never streaming
        }
    }

    /// Get the cached tokenizer, cloning the Arc (cheap 8-byte clone)
    ///
    /// Returns None if tokenizer hasn't been resolved yet.
    /// The tokenizer is resolved once in the preparation stage and cached for reuse.
    pub fn tokenizer_arc(&self) -> Option<Arc<dyn Tokenizer>> {
        self.state.tokenizer.clone()
    }
}

/// Some methods are kept for API completeness even if currently unused.
#[allow(dead_code)]
impl WorkerSelection {
    pub fn is_dual(&self) -> bool {
        matches!(self, Self::Dual { .. })
    }

    pub fn single(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Single { worker } => Some(worker),
            _ => None,
        }
    }

    /// Record circuit breaker outcome for all workers
    pub fn record_outcome(&self, success: bool) {
        match self {
            Self::Single { worker } => worker.record_outcome(success),
            Self::Dual { prefill, decode } => {
                prefill.record_outcome(success);
                decode.record_outcome(success);
            }
        }
    }

    /// Record circuit breaker outcomes for dual dispatch (individual tracking)
    pub fn record_dual_outcomes(&self, prefill_success: bool, decode_success: bool) {
        if let Self::Dual { prefill, decode } = self {
            prefill.record_outcome(prefill_success);
            decode.record_outcome(decode_success);
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn dual(&self) -> Option<(&Arc<dyn Worker>, &Arc<dyn Worker>)> {
        match self {
            Self::Dual { prefill, decode } => Some((prefill, decode)),
            _ => None,
        }
    }

    pub fn prefill_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            _ => None,
        }
    }

    pub fn decode_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            _ => None,
        }
    }
}

/// Some methods are kept for API completeness even if currently unused.
#[allow(dead_code)]
impl ClientSelection {
    pub fn single(&self) -> Option<&GrpcClient> {
        match self {
            Self::Single { client } => Some(client),
            _ => None,
        }
    }

    pub fn single_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Single { client } => Some(client),
            _ => None,
        }
    }

    pub fn dual_mut(&mut self) -> Option<(&mut GrpcClient, &mut GrpcClient)> {
        match self {
            Self::Dual { prefill, decode } => Some((prefill, decode)),
            _ => None,
        }
    }

    pub fn prefill_client(&self) -> Option<&GrpcClient> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            _ => None,
        }
    }

    pub fn prefill_client_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            _ => None,
        }
    }

    pub fn decode_client(&self) -> Option<&GrpcClient> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            _ => None,
        }
    }

    pub fn decode_client_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            _ => None,
        }
    }
}

/// Result of request execution (streams from workers)
/// Uses ProtoStream to automatically abort on cancellation
pub(crate) enum ExecutionResult {
    Single {
        stream: ProtoStream,
    },
    Dual {
        prefill: ProtoStream,
        decode: Box<ProtoStream>,
    },
    /// Embedding requests return a single response, not a stream
    Embedding {
        response: ProtoEmbedComplete,
    },
}

/// Final processed response
#[derive(Debug)]
pub(crate) enum FinalResponse {
    Chat(ChatCompletionResponse),
    /// Generate response is a Vec of GenerateResponse (n=1 returns single item, n>1 returns multiple)
    Generate(Vec<GenerateResponse>),
    /// Embedding response
    Embedding(EmbeddingResponse),
    /// Classification response
    Classify(ClassifyResponse),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};
    use crate::policies::{PrefillAwarePolicy, SelectWorkerInfo};

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .build(),
        )
    }

    /// Dropping the guards must close the `prefill_aware` dispatch, otherwise
    /// the per-worker ledger only shrinks through its TTL.
    #[tokio::test]
    async fn guards_close_prefill_aware_dispatches() {
        let policy: Arc<dyn LoadBalancingPolicy> = Arc::new(PrefillAwarePolicy::new());
        let prefill = worker("http://prefill:8000");
        let decode = worker("http://decode:8000");
        let (prefill_url, decode_url) = (prefill.url().to_string(), decode.url().to_string());

        // Dispatch one request to each worker so both ledgers hold 50 chars.
        // Distinct prompts: the tree is cluster-wide, so a repeated prompt would
        // match fully and account nothing.
        for (w, seed) in [
            (prefill.clone(), "x".repeat(50)),
            (decode.clone(), "y".repeat(50)),
        ] {
            let info = SelectWorkerInfo {
                request_text: Some(&seed),
                ..Default::default()
            };
            policy.select_worker(&[w], &info).await.unwrap();
        }
        let pa = policy
            .as_any()
            .downcast_ref::<PrefillAwarePolicy>()
            .unwrap();
        assert_eq!(pa.inflight_chars(&prefill_url), 50);
        assert_eq!(pa.inflight_chars(&decode_url), 50);

        // Single selection releases only the worker it was attached to.
        {
            let _guard = LoadGuards::new(
                &WorkerSelection::Single {
                    worker: prefill.clone(),
                },
                None,
                Some(&policy),
                (None, None),
            );
        }
        assert_eq!(pa.inflight_chars(&prefill_url), 0);
        assert_eq!(pa.inflight_chars(&decode_url), 50);

        // Re-seed prefill with a prompt nobody has used: the tree is
        // cluster-wide, so any previously inserted prompt matches fully and
        // would account zero characters.
        let fresh = "z".repeat(50);
        let info = SelectWorkerInfo {
            request_text: Some(&fresh),
            ..Default::default()
        };
        policy
            .select_worker(std::slice::from_ref(&prefill), &info)
            .await
            .unwrap();
        assert_eq!(pa.inflight_chars(&prefill_url), 50);
        {
            let _guards = LoadGuards::new(
                &WorkerSelection::Dual {
                    prefill: prefill.clone(),
                    decode: decode.clone(),
                },
                None,
                Some(&policy),
                (None, None),
            );
        }
        assert_eq!(pa.inflight_chars(&prefill_url), 0);
        assert_eq!(pa.inflight_chars(&decode_url), 0);
    }

    /// Without a policy (or with a non-prefill_aware one) the guards must still
    /// manage the worker's own load counter.
    #[tokio::test]
    async fn guards_without_policy_are_plain_load_guards() {
        let w = worker("http://w:8000");
        {
            let _guards = LoadGuards::new(
                &WorkerSelection::Single { worker: w.clone() },
                None,
                None,
                (None, None),
            );
            assert_eq!(w.load(), 1);
        }
        assert_eq!(w.load(), 0);
    }
}
