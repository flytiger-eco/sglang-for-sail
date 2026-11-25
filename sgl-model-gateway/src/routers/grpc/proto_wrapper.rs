//! Protocol buffer type wrappers for SGLang and vLLM backends
//!
//! This module provides unified enums that wrap proto types from both SGLang and vLLM,
//! allowing the router to work with either backend transparently.

use futures_util::StreamExt;
use smg_grpc_client::{
    sglang_proto::{self as sglang, generate_complete::MatchedStop},
    sglang_scheduler::AbortOnDropStream as SglangStream,
    vllm_engine::AbortOnDropStream as VllmStream,
    vllm_proto as vllm,
};

/// Unified ProtoRequest
#[derive(Clone)]
pub enum ProtoRequest {
    Generate(ProtoGenerateRequest),
    Embed(ProtoEmbedRequest),
}

impl ProtoRequest {
    /// Get request ID from either variant
    pub fn request_id(&self) -> &str {
        match self {
            Self::Generate(req) => req.request_id(),
            Self::Embed(req) => req.request_id(),
        }
    }
}

/// Unified GenerateRequest that works with both backends
#[derive(Clone)]
pub enum ProtoGenerateRequest {
    Sglang(Box<sglang::GenerateRequest>),
    Vllm(Box<vllm::GenerateRequest>),
}

impl ProtoGenerateRequest {
    /// Get SGLang variant (panics if vLLM)
    pub fn as_sglang(&self) -> &sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            Self::Vllm(_) => panic!("Expected SGLang GenerateRequest, got vLLM"),
        }
    }

    /// Get mutable SGLang variant (panics if vLLM)
    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            Self::Vllm(_) => panic!("Expected SGLang GenerateRequest, got vLLM"),
        }
    }

    /// Get vLLM variant (panics if SGLang)
    pub fn as_vllm(&self) -> &vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            Self::Sglang(_) => panic!("Expected vLLM GenerateRequest, got SGLang"),
        }
    }

    /// Get mutable vLLM variant (panics if SGLang)
    pub fn as_vllm_mut(&mut self) -> &mut vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            Self::Sglang(_) => panic!("Expected vLLM GenerateRequest, got SGLang"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Clone the inner request (for passing to generate())
    pub fn clone_inner(&self) -> Self {
        self.clone()
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
            Self::Vllm(req) => &req.request_id,
        }
    }
}

/// Unified GenerateResponse from stream
pub enum ProtoGenerateResponse {
    Sglang(Box<sglang::GenerateResponse>),
    Vllm(vllm::GenerateResponse),
}

impl ProtoGenerateResponse {
    /// Get the response variant (chunk, complete, or error)
    ///
    /// Consumes self to avoid cloning large proto messages in hot streaming path
    pub fn into_response(self) -> ProtoResponseVariant {
        match self {
            Self::Sglang(resp) => match resp.response {
                Some(sglang::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Sglang(chunk))
                }
                Some(sglang::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Sglang(complete))
                }
                Some(sglang::generate_response::Response::Error(error)) => {
                    ProtoResponseVariant::Error(ProtoGenerateError::Sglang(error))
                }
                None => ProtoResponseVariant::None,
            },
            Self::Vllm(resp) => match resp.response {
                Some(vllm::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Vllm(chunk))
                }
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Vllm(complete))
                }
                // Note: vLLM proto no longer has Error variant in GenerateResponse
                None => ProtoResponseVariant::None,
            },
        }
    }
}

/// Response variant extracted from GenerateResponse
pub enum ProtoResponseVariant {
    Chunk(ProtoGenerateStreamChunk),
    Complete(ProtoGenerateComplete),
    Error(ProtoGenerateError),
    None,
}

/// Unified GenerateStreamChunk
#[derive(Clone)]
pub enum ProtoGenerateStreamChunk {
    Sglang(sglang::GenerateStreamChunk),
    Vllm(vllm::GenerateStreamChunk),
}

impl ProtoGenerateStreamChunk {
    /// Get SGLang variant (panics if vLLM)
    pub fn as_sglang(&self) -> &sglang::GenerateStreamChunk {
        match self {
            Self::Sglang(chunk) => chunk,
            Self::Vllm(_) => panic!("Expected SGLang GenerateStreamChunk, got vLLM"),
        }
    }

    /// Get vLLM variant (panics if SGLang)
    pub fn as_vllm(&self) -> &vllm::GenerateStreamChunk {
        match self {
            Self::Vllm(chunk) => chunk,
            Self::Sglang(_) => panic!("Expected vLLM GenerateStreamChunk, got SGLang"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Get token IDs from chunk (common field)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.token_ids,
            Self::Vllm(c) => &c.token_ids,
        }
    }

    /// Get index (for n>1 support)
    /// vLLM doesn't support n>1, so always returns 0
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(_) => 0, // vLLM doesn't support n>1
        }
    }

    /// Get output logprobs (SGLang only, returns None for vLLM)
    pub fn output_logprobs(&self) -> Option<&sglang::OutputLogProbs> {
        match self {
            Self::Sglang(c) => c.output_logprobs.as_ref(),
            Self::Vllm(_) => None, // TODO: vLLM logprobs mapping
        }
    }

    /// Get prompt tokens (cumulative)
    pub fn prompt_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens as i32,
        }
    }

    /// Get completion tokens (cumulative)
    pub fn completion_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens as i32,
        }
    }

    /// Get cached tokens (cumulative)
    pub fn cached_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens as i32,
        }
    }
}

/// Unified GenerateComplete response
#[derive(Clone)]
pub enum ProtoGenerateComplete {
    Sglang(sglang::GenerateComplete),
    Vllm(vllm::GenerateComplete),
}

impl ProtoGenerateComplete {
    /// Get SGLang variant (panics if vLLM)
    pub fn as_sglang(&self) -> &sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            Self::Vllm(_) => panic!("Expected SGLang GenerateComplete, got vLLM"),
        }
    }

    /// Get mutable SGLang variant (panics if vLLM)
    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            Self::Vllm(_) => panic!("Expected SGLang GenerateComplete, got vLLM"),
        }
    }

    /// Get vLLM variant (panics if SGLang)
    pub fn as_vllm(&self) -> &vllm::GenerateComplete {
        match self {
            Self::Vllm(complete) => complete,
            Self::Sglang(_) => panic!("Expected vLLM GenerateComplete, got SGLang"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Get token IDs from either backend (output_ids in proto)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
        }
    }

    /// Get prompt tokens
    pub fn prompt_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens as i32,
        }
    }

    /// Get completion tokens
    pub fn completion_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens as i32,
        }
    }

    /// Get finish reason
    pub fn finish_reason(&self) -> &str {
        match self {
            Self::Sglang(c) => &c.finish_reason,
            Self::Vllm(c) => &c.finish_reason,
        }
    }

    /// Get index (for n>1 support)
    /// vLLM doesn't support n>1, so always returns 0
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(_) => 0, // vLLM doesn't have index field (n>1 not supported)
        }
    }

    /// Get matched stop (SGLang only, returns oneof)
    /// vLLM doesn't have matched_stop, returns None
    pub fn matched_stop(&self) -> Option<&MatchedStop> {
        match self {
            Self::Sglang(c) => c.matched_stop.as_ref(),
            Self::Vllm(_) => None, // vLLM doesn't have matched_stop
        }
    }

    /// Get output IDs (decode tokens only)
    pub fn output_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
        }
    }

    /// Get cached tokens
    pub fn cached_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens as i32,
        }
    }

    /// Get input logprobs (SGLang only)
    pub fn input_logprobs(&self) -> Option<&sglang::InputLogProbs> {
        match self {
            Self::Sglang(c) => c.input_logprobs.as_ref(),
            Self::Vllm(_) => None, // vLLM doesn't have input_logprobs
        }
    }

    /// Get output logprobs
    pub fn output_logprobs(&self) -> Option<&sglang::OutputLogProbs> {
        match self {
            Self::Sglang(c) => c.output_logprobs.as_ref(),
            Self::Vllm(_) => None, // TODO: vLLM logprobs mapping
        }
    }
}

/// Unified GenerateError
/// Note: vLLM proto no longer has GenerateError - errors are returned via gRPC status
#[derive(Clone)]
pub enum ProtoGenerateError {
    Sglang(sglang::GenerateError),
}

impl ProtoGenerateError {
    /// Get error message
    pub fn message(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.message,
        }
    }
}

/// Unified stream wrapper
pub enum ProtoStream {
    Sglang(SglangStream),
    Vllm(VllmStream),
    #[cfg(test)]
    Mock(futures_util::stream::BoxStream<'static, Result<ProtoGenerateResponse, tonic::Status>>),
    Tracked {
        stream: Box<ProtoStream>,
        guard: Option<crate::core::WorkerLoadGuard>,
    },
}

impl ProtoStream {
    pub fn with_guard(self, guard: crate::core::WorkerLoadGuard) -> Self {
        Self::Tracked {
            stream: Box::new(self),
            guard: Some(guard),
        }
    }

    /// Get next item from stream
    pub async fn next(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
        match self {
            #[cfg(test)]
            Self::Mock(stream) => stream.next().await,
            Self::Tracked { stream, guard } => {
                let item = Box::pin(stream.next()).await;
                if let Some(guard) = guard.as_mut() {
                    guard.on_first_byte();
                    if matches!(item, Some(Err(_))) {
                        guard.set_success(false);
                    }
                }
                if item.is_none() || matches!(item, Some(Err(_))) {
                    guard.take();
                }
                item
            }
            Self::Sglang(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Sglang(Box::new(r)))),
            Self::Vllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(ProtoGenerateResponse::Vllm)),
        }
    }

    /// Mark stream as completed (no abort needed)
    pub fn mark_completed(&mut self) {
        match self {
            #[cfg(test)]
            Self::Mock(_) => {}
            Self::Tracked { stream, guard } => {
                stream.mark_completed();
                guard.take();
            }
            Self::Sglang(stream) => stream.mark_completed(),
            Self::Vllm(stream) => stream.mark_completed(),
        }
    }
}

/// Unified EmbedRequest that works with both backends
#[derive(Clone)]
pub enum ProtoEmbedRequest {
    Sglang(Box<sglang::EmbedRequest>),
}

impl ProtoEmbedRequest {
    /// Get SGLang variant
    pub fn as_sglang(&self) -> &sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
        }
    }

    /// Get mutable SGLang variant
    pub fn as_sglang_mut(&mut self) -> &mut sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Clone the inner request (for passing to embed())
    pub fn clone_inner(&self) -> Self {
        self.clone()
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
        }
    }
}

/// Unified EmbedResponse
pub enum ProtoEmbedResponse {
    Sglang(sglang::EmbedResponse),
}

impl ProtoEmbedResponse {
    /// Get the response variant (complete or error)
    pub fn into_response(self) -> ProtoEmbedResponseVariant {
        match self {
            Self::Sglang(resp) => match resp.response {
                Some(sglang::embed_response::Response::Complete(complete)) => {
                    ProtoEmbedResponseVariant::Complete(ProtoEmbedComplete::Sglang(complete))
                }
                Some(sglang::embed_response::Response::Error(error)) => {
                    ProtoEmbedResponseVariant::Error(ProtoEmbedError::Sglang(error))
                }
                None => ProtoEmbedResponseVariant::None,
            },
        }
    }
}

/// Response variant extracted from EmbedResponse
pub enum ProtoEmbedResponseVariant {
    Complete(ProtoEmbedComplete),
    Error(ProtoEmbedError),
    None,
}

/// Unified EmbedComplete response
#[derive(Clone)]
pub enum ProtoEmbedComplete {
    Sglang(sglang::EmbedComplete),
}

impl ProtoEmbedComplete {
    /// Get embeddings
    pub fn embedding(&self) -> &[f32] {
        match self {
            Self::Sglang(c) => &c.embedding,
        }
    }

    /// Get prompt tokens
    pub fn prompt_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
        }
    }

    /// Get cached tokens
    pub fn cached_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
        }
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.embedding_dim,
        }
    }
}

/// Unified EmbedError
#[derive(Clone)]
pub enum ProtoEmbedError {
    Sglang(sglang::EmbedError),
}

impl ProtoEmbedError {
    /// Get error message
    pub fn message(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.message,
        }
    }

    /// Get error code
    pub fn code(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.code,
        }
    }
}

#[cfg(test)]
mod reservation_tests {
    use super::*;
    use crate::{
        core::{BasicWorkerBuilder, Worker, WorkerLoadGuard},
        policies::{LoadBalancingPolicy, PrefillAwarePolicy, SelectWorkerInfo},
    };
    use std::sync::Arc;

    #[tokio::test]
    async fn grpc_first_chunk_transitions_once_and_each_stream_settles_independently() {
        let policy: Arc<dyn LoadBalancingPolicy> = Arc::new(PrefillAwarePolicy::new());
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://worker").build());
        let pa = policy
            .as_any()
            .downcast_ref::<PrefillAwarePolicy>()
            .unwrap();
        let mut streams = Vec::new();
        for text in ["prefill", "decode"] {
            let (_, id) = policy
                .select_worker_with_token(
                    std::slice::from_ref(&worker),
                    &SelectWorkerInfo {
                        request_text: Some(text),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            let guard = WorkerLoadGuard::with_policy(worker.clone(), None, &policy, id);
            let items = [11, 22].map(|id| {
                Ok(ProtoGenerateResponse::Sglang(Box::new(
                    sglang::GenerateResponse {
                        request_id: "test".into(),
                        response: Some(sglang::generate_response::Response::Chunk(
                            sglang::GenerateStreamChunk {
                                token_ids: vec![id],
                                ..Default::default()
                            },
                        )),
                    },
                )))
            });
            streams.push(
                ProtoStream::Mock(futures_util::stream::iter(items).boxed()).with_guard(guard),
            );
        }
        let mut decode = streams.pop().unwrap();
        let mut prefill = streams.pop().unwrap();
        assert_eq!(pa.phase_counts(worker.url()), (2, 0));
        for id in [11, 22] {
            let response = prefill.next().await.unwrap().unwrap();
            let ProtoResponseVariant::Chunk(chunk) = response.into_response() else {
                panic!("expected chunk")
            };
            assert_eq!(chunk.token_ids(), &[id]);
            assert_eq!(pa.phase_counts(worker.url()), (1, 1));
        }
        assert!(prefill.next().await.is_none());
        assert_eq!(pa.phase_counts(worker.url()), (1, 0));
        assert_eq!(pa.inflight_chars(worker.url()), 6);
        decode.next().await.unwrap().unwrap();
        assert_eq!(pa.phase_counts(worker.url()), (0, 1));
        drop(decode); // cancellation before EOF
        assert_eq!(pa.phase_counts(worker.url()), (0, 0));
        assert_eq!(worker.load(), 0);
    }

    #[derive(Debug, Default)]
    struct HookCounts {
        first: std::sync::atomic::AtomicUsize,
        completed: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl LoadBalancingPolicy for HookCounts {
        async fn select_worker(
            &self,
            _: &[Arc<dyn Worker>],
            _: &SelectWorkerInfo<'_>,
        ) -> Option<usize> {
            None
        }
        fn name(&self) -> &'static str {
            "prefill_aware"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn on_first_token(&self, _: &str, id: Option<u64>) {
            assert_eq!(id, Some(7));
            self.first.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        fn on_dispatch_complete(&self, _: &str, id: Option<u64>, success: bool) {
            assert_eq!(id, Some(7));
            assert!(!success);
            self.completed
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn grpc_error_first_item_signals_once_and_releases_guard() {
        use std::sync::atomic::Ordering;
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://worker").build());
        let counts = Arc::new(HookCounts::default());
        let policy: Arc<dyn LoadBalancingPolicy> = counts.clone();
        let guard = WorkerLoadGuard::with_policy(worker.clone(), None, &policy, Some(7));
        let mut stream = ProtoStream::Mock(
            futures_util::stream::iter([Err(tonic::Status::internal("upstream error"))]).boxed(),
        )
        .with_guard(guard);
        assert_eq!(
            stream.next().await.unwrap().err().unwrap().message(),
            "upstream error"
        );
        assert_eq!(worker.load(), 0);
        assert!(stream.next().await.is_none());
        drop(stream);
        assert_eq!(counts.first.load(Ordering::SeqCst), 1);
        assert_eq!(counts.completed.load(Ordering::SeqCst), 1);
    }
}
