//! Streaming PD handoff after actual decode output, with prefill error handling.

use std::sync::Arc;

use axum::response::Response;
use futures_util::StreamExt;
use serde_json::Value;
use tracing::{warn, Instrument};

use super::{error_type_from_status, BreakerOutcomesRecorded, PDRequestContext, PDRouter};
use crate::{
    core::{Worker, WorkerLoadGuard},
    observability::{
        events::{self, Event},
        metrics::{metrics_labels, Metrics},
    },
    routers::{error, header_utils},
};

/// Detect a generated token in complete SSE events, across arbitrary HTTP chunks.
/// Headers, role-only deltas, keepalives, errors and [DONE] do not open the gate.
#[derive(Default)]
struct DecodeGenerationProbe {
    pending: Vec<u8>,
}

impl DecodeGenerationProbe {
    fn feed(&mut self, chunk: &[u8]) -> bool {
        self.pending.extend_from_slice(chunk);
        while let Some((end, delimiter_len)) = self
            .pending
            .windows(2)
            .position(|v| v == b"\n\n")
            .map(|end| (end, 2))
            .into_iter()
            .chain(
                self.pending
                    .windows(4)
                    .position(|v| v == b"\r\n\r\n")
                    .map(|end| (end, 4)),
            )
            .min_by_key(|(end, _)| *end)
        {
            let event = String::from_utf8_lossy(&self.pending[..end]);
            let data = event
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim_start)
                .collect::<Vec<_>>()
                .join("\n");
            let generated =
                serde_json::from_str::<Value>(&data).is_ok_and(|value| Self::has_output(&value));
            self.pending.drain(..end + delimiter_len);
            if generated {
                return true;
            }
        }
        false
    }

    fn has_output(value: &Value) -> bool {
        if value.get("error").is_some_and(|error| !error.is_null()) {
            return false;
        }
        let nonempty = |value: Option<&Value>| {
            value
                .and_then(Value::as_str)
                .is_some_and(|text| !text.is_empty())
        };
        if nonempty(value.get("text"))
            || value
                .get("output_ids")
                .and_then(Value::as_array)
                .is_some_and(|ids| !ids.is_empty())
            || value
                .pointer("/meta_info/completion_tokens")
                .and_then(Value::as_u64)
                .is_some_and(|tokens| tokens > 0)
        {
            return true;
        }
        value
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| {
                choices.iter().any(|choice| {
                    if nonempty(choice.get("text")) {
                        return true;
                    }
                    let delta = &choice["delta"];
                    ["content", "reasoning_content", "reasoning"]
                        .iter()
                        .any(|key| nonempty(delta.get(*key)))
                        || delta
                            .get("tool_calls")
                            .and_then(Value::as_array)
                            .is_some_and(|calls| {
                                calls.iter().any(|call| {
                                    nonempty(call.pointer("/function/name"))
                                        || nonempty(call.pointer("/function/arguments"))
                                })
                            })
                })
            })
    }
}

impl PDRouter {
    /// Race prefill completion against *generated SSE data*, not decode headers.
    /// Until either succeeds, a prefill failure still cancels the paired decode.
    pub(super) async fn execute_streaming_until_decode_token(
        &self,
        prefill_request: reqwest::RequestBuilder,
        decode_request: reqwest::RequestBuilder,
        context: &PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        admission_guards: [WorkerLoadGuard; 2],
    ) -> Response {
        let [prefill_guard, decode_guard] = admission_guards;
        let prefill_for_completion = Arc::clone(&prefill);
        let mut prefill_completion = Box::pin(async move {
            let result = prefill_request.send().await;
            let worker_ok = match &result {
                Ok(response) => {
                    response.status().is_success() || response.status().is_client_error()
                }
                Err(_) => false,
            };
            let result =
                Self::process_prefill_response(result, prefill_for_completion.url(), false).await;
            prefill_for_completion.record_outcome(worker_ok);
            result.map(|_| ())
        });
        let decode_request = decode_request.send();
        tokio::pin!(decode_request);
        let mut prefill_done = false;

        let decode_result = loop {
            tokio::select! {
                biased;
                result = &mut prefill_completion, if !prefill_done => {
                    prefill_done = true;
                    if let Err(mut response) = result {
                        response.extensions_mut().insert(BreakerOutcomesRecorded);
                        return response;
                    }
                }
                result = &mut decode_request => break result,
            }
        };
        let response = match decode_result {
            Ok(response) => response,
            Err(error) => {
                decode.record_outcome(false);
                let mut response = error::bad_gateway(
                    "decode_server_error",
                    format!("Decode server error: {error}"),
                );
                response.extensions_mut().insert(BreakerOutcomesRecorded);
                return response;
            }
        };
        if !response.status().is_success() {
            // A rejected decode cannot use prefill's KV. Drop the pending P
            // future and preserve the existing decode error SSE/status logic.
            let mut response = self
                .handle_decode_error_response(response, context, prefill, decode)
                .await;
            response.extensions_mut().insert(BreakerOutcomesRecorded);
            return response;
        }

        let status = response.status();
        let response_headers = header_utils::preserve_response_headers(response.headers());
        let mut stream = response.bytes_stream().fuse().boxed();
        let mut buffered = Vec::new();
        let mut buffered_bytes = 0;
        let mut generated = false;
        let mut probe = DecodeGenerationProbe::default();

        // Keep pre-token buffering bounded. An unusual stream with >1MiB of
        // metadata/keepalives falls back to the original prefill completion gate.
        while !prefill_done && !generated && buffered_bytes < 1024 * 1024 {
            tokio::select! {
                biased;
                result = &mut prefill_completion => {
                    prefill_done = true;
                    if let Err(mut response) = result {
                        response.extensions_mut().insert(BreakerOutcomesRecorded);
                        return response;
                    }
                }
                chunk = stream.next() => {
                    match chunk {
                        Some(Ok(chunk)) => {
                            buffered_bytes += chunk.len();
                            generated = probe.feed(&chunk);
                            buffered.push(Ok(chunk));
                        }
                        Some(Err(error)) => {
                            decode.record_outcome(false);
                            let mut response = error::bad_gateway(
                                "decode_stream_error", format!("Decode stream error: {error}"),
                            );
                            response.extensions_mut().insert(BreakerOutcomesRecorded);
                            return response;
                        }
                        None => break,
                    }
                }
            }
        }

        if !prefill_done && !generated {
            // Empty/role-only streams are not evidence of successful KV handoff.
            if let Err(mut response) = (&mut prefill_completion).await {
                response.extensions_mut().insert(BreakerOutcomesRecorded);
                return response;
            }
            prefill_done = true;
        }

        let guards = if prefill_done {
            vec![prefill_guard, decode_guard]
        } else {
            // D has generated real output. Finish reading P independently so
            // delayed cleanup cannot hold up the client. Retain P's admission
            // guard until its response completes and record its real outcome.
            let prefill_url = prefill.url().to_string();
            // The shared HTTP client's total request timeout also bounds this
            // task, including response-body reads (AppContextBuilder::with_client).
            tokio::spawn(
                async move {
                    let _guard = prefill_guard;
                    if let Err(response) = prefill_completion.await {
                        let status = response.status();
                        warn!(prefill_url, %status,
                        "Prefill HTTP response failed after decode generated output");
                        if status.is_server_error() {
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_PREFILL,
                                metrics_labels::CONNECTION_HTTP,
                                error_type_from_status(status),
                            );
                        }
                    }
                }
                .instrument(tracing::Span::current()),
            );
            vec![decode_guard]
        };

        events::RequestReceivedEvent {}.emit();
        let replay = futures_util::stream::iter(buffered).chain(stream);
        let mut response = self.create_streaming_response(
            replay,
            status,
            None,
            false,
            Some(response_headers),
            prefill,
            decode,
            Some(guards),
        );
        // P is accounted by its completion future, D by BreakerTrackedStream.
        response.extensions_mut().insert(BreakerOutcomesRecorded);
        response
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use axum::{body::Body, http::StatusCode, routing::post, Router};
    use futures_util::StreamExt;
    use tokio::{net::TcpListener, sync::Notify, task::JoinHandle, time::timeout};

    use super::{DecodeGenerationProbe, PDRequestContext, PDRouter};
    use crate::{
        config::{PolicyConfig, RetryConfig},
        core::{BasicWorkerBuilder, Worker, WorkerLoadGuard, WorkerRegistry, WorkerType},
        policies::PolicyRegistry,
    };

    fn router() -> PDRouter {
        PDRouter {
            worker_registry: Arc::new(WorkerRegistry::new()),
            policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            client: reqwest::Client::new(),
            retry_config: RetryConfig::default(),
            api_key: None,
            enable_igw: false,
        }
    }

    fn context() -> PDRequestContext<'static> {
        PDRequestContext {
            route: "/v1/chat/completions",
            batch_size: None,
            is_stream: true,
            return_logprob: false,
            request_text: None,
            model_id: None,
            headers: None,
        }
    }

    fn worker(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .build(),
        )
    }

    async fn serve(app: Router) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, task)
    }

    #[tokio::test]
    async fn generated_token_opens_stream_before_prefill_response() {
        check_generated_handoff(false).await;
    }

    #[tokio::test]
    async fn delayed_prefill_request_timeout_releases_load() {
        check_generated_handoff(true).await;
    }

    async fn check_generated_handoff(timeout_prefill: bool) {
        let release = Arc::new(Notify::new());
        let gate = Arc::clone(&release);
        let (p_url, p_task) = serve(Router::new().route(
            "/",
            post(move || {
                let gate = Arc::clone(&gate);
                async move {
                    gate.notified().await;
                    (StatusCode::INTERNAL_SERVER_ERROR, "injected P failure")
                }
            }),
        ))
        .await;
        let wire = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\ndata: {\"error\":null,\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n";
        let (d_url, d_task) = serve(Router::new().route(
            "/",
            post(move || async move { ([("content-type", "text/event-stream")], wire) }),
        ))
        .await;
        let mut router = router();
        if timeout_prefill {
            router.client = reqwest::Client::builder()
                .timeout(Duration::from_secs(1))
                .build()
                .unwrap();
        }
        let prefill = worker(
            &p_url,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        );
        let decode = worker(&d_url, WorkerType::Decode);
        let guards = [
            WorkerLoadGuard::new(Arc::clone(&prefill), None),
            WorkerLoadGuard::new(Arc::clone(&decode), None),
        ];
        let response = timeout(
            Duration::from_secs(2),
            router.execute_streaming_until_decode_token(
                router.client.post(&p_url),
                router.client.post(&d_url),
                &context(),
                Arc::clone(&prefill),
                Arc::clone(&decode),
                guards,
            ),
        )
        .await
        .expect("D output must not wait for the blocked P response");
        assert_eq!(response.status(), StatusCode::OK);
        assert!(!response.headers().contains_key("content-length"));
        let body = timeout(
            Duration::from_secs(2),
            axum::body::to_bytes(response.into_body(), usize::MAX),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(body.as_ref(), wire.as_bytes());
        assert_eq!(decode.load(), 0);
        assert_eq!(
            prefill.load(),
            1,
            "P stays counted until its response finishes"
        );
        assert_eq!(decode.circuit_breaker().total_successes(), 1);
        if !timeout_prefill {
            release.notify_one();
        }
        timeout(Duration::from_secs(3), async {
            while prefill.load() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("P completion must release its admission guard");
        assert_eq!(prefill.circuit_breaker().total_failures(), 1);
        p_task.abort();
        d_task.abort();
    }

    #[tokio::test]
    async fn role_only_stream_still_propagates_prefill_failure() {
        let release = Arc::new(Notify::new());
        let gate = Arc::clone(&release);
        let (p_url, p_task) = serve(Router::new().route(
            "/",
            post(move || {
                let gate = Arc::clone(&gate);
                async move {
                    gate.notified().await;
                    (StatusCode::INTERNAL_SERVER_ERROR, "injected P failure")
                }
            }),
        ))
        .await;
        let d_seen = Arc::new(Notify::new());
        let seen = Arc::clone(&d_seen);
        let (d_url, d_task) = serve(Router::new().route("/", post(move || {
            let seen = Arc::clone(&seen);
            async move {
                seen.notify_one();
                let role = bytes::Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n");
                let stream = futures_util::stream::once(async move {
                    Ok::<_, std::convert::Infallible>(role)
                }).chain(futures_util::stream::pending());
                axum::response::Response::builder().header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream)).unwrap()
            }
        }))).await;
        let router = router();
        let prefill = worker(
            &p_url,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        );
        let decode = worker(&d_url, WorkerType::Decode);
        let guards = [
            WorkerLoadGuard::new(Arc::clone(&prefill), None),
            WorkerLoadGuard::new(Arc::clone(&decode), None),
        ];
        let ctx = context();
        let mut response = Box::pin(router.execute_streaming_until_decode_token(
            router.client.post(&p_url),
            router.client.post(&d_url),
            &ctx,
            Arc::clone(&prefill),
            Arc::clone(&decode),
            guards,
        ));
        timeout(Duration::from_secs(2), async {
            tokio::select! {
                _ = d_seen.notified() => {},
                _ = &mut response => panic!("headers alone must not open the stream"),
            }
        })
        .await
        .unwrap();
        assert!(timeout(Duration::from_millis(50), &mut response)
            .await
            .is_err());
        release.notify_one();
        let response = timeout(Duration::from_secs(2), response).await.unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(prefill.load(), 0);
        assert_eq!(decode.load(), 0);
        assert_eq!(prefill.circuit_breaker().total_failures(), 1);
        assert_eq!(decode.circuit_breaker().total_failures(), 0);
        assert_eq!(decode.circuit_breaker().total_successes(), 0);
        p_task.abort();
        d_task.abort();
    }

    #[test]
    fn generated_token_probe_ignores_metadata_and_preserves_chunk_boundaries() {
        let prefix = b": keepalive\n\ndata: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n";
        let token = b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n";
        for width in [1, 3, 17, 4096] {
            let mut probe = DecodeGenerationProbe::default();
            for chunk in prefix.chunks(width) {
                assert!(!probe.feed(chunk));
            }
            assert!(token.chunks(width).any(|chunk| probe.feed(chunk)));
        }
        for wire in [
            "data: [DONE]\n\n",
            "data: {\"error\":{\"message\":\"bad\"}}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}]}\n\n",
        ] {
            assert!(!DecodeGenerationProbe::default().feed(wire.as_bytes()));
        }
        for wire in [
            "data: {\"text\":\"x\"}\r\n\r\n",
            "data: {\"choices\":[{\"text\":\"x\"}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"x\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{\"}}]}}]}\n\n",
            "data: {\"output_ids\":[12]}\n\n",
        ] {
            assert!(DecodeGenerationProbe::default().feed(wire.as_bytes()));
        }
    }
}
