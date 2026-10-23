use super::*;

use crate::utils::strip_think_tag;

use anyhow::{bail, Context, Result};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{json, Value};

const API_BASE: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct OpenAIConfig {
    pub name: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub organization_id: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl OpenAIClient {
    config_get_fn!(api_key, get_api_key);
    config_get_fn!(api_base, get_api_base);

    pub const PROMPTS: [PromptAction<'static>; 1] = [("api_key", "API Key", None)];
}

impl_client_trait!(
    OpenAIClient,
    (
        prepare_chat_completions,
        openai_chat_completions,
        openai_chat_completions_streaming
    ),
    (prepare_embeddings, openai_embeddings),
    (noop_prepare_rerank, noop_rerank),
);

fn prepare_chat_completions(
    self_: &OpenAIClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base = self_
        .get_api_base()
        .unwrap_or_else(|_| API_BASE.to_string());

    let url = format!("{}/chat/completions", api_base.trim_end_matches('/'));

    let body = openai_build_chat_completions_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);
    if let Some(organization_id) = &self_.config.organization_id {
        request_data.header("OpenAI-Organization", organization_id);
    }

    Ok(request_data)
}

fn prepare_embeddings(self_: &OpenAIClient, data: &EmbeddingsData) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base = self_
        .get_api_base()
        .unwrap_or_else(|_| API_BASE.to_string());

    let url = format!("{api_base}/embeddings");

    let body = openai_build_embeddings_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);
    if let Some(organization_id) = &self_.config.organization_id {
        request_data.header("OpenAI-Organization", organization_id);
    }

    Ok(request_data)
}

pub async fn openai_chat_completions(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<ChatCompletionsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }

    debug!("non-stream-data: {data}");
    openai_extract_chat_completions(&data)
}

#[derive(Default)]
struct OpenAiChatStreamState {
    tool_calls: Vec<OpenAiStreamToolCall>,
    current_index: Option<u64>,
    reasoning_state: i32,
}

#[derive(Default)]
struct OpenAiStreamToolCall {
    index: u64,
    function_name: String,
    function_arguments: String,
    function_id: String,
}

pub async fn openai_chat_completions_streaming(
    builder: RequestBuilder,
    handler: &mut SseHandler,
    _model: &Model,
) -> Result<()> {
    let mut state = OpenAiChatStreamState::default();
    sse_stream(builder, |message| {
        handle_openai_stream_message(&mut state, handler, message)
    })
    .await
}

fn handle_openai_stream_message(
    state: &mut OpenAiChatStreamState,
    handler: &mut SseHandler,
    message: SseMmessage,
) -> Result<bool> {
    if message.data == "[DONE]" {
        for call in &mut state.tool_calls {
            flush_openai_stream_tool_call(call, handler)?;
        }
        return Ok(true);
    }
    let data: Value = serde_json::from_str(&message.data)?;
    debug!("stream-data: {data}");
    if let Some(text) = data["choices"][0]["delta"]["content"]
        .as_str()
        .filter(|v| !v.is_empty())
    {
        if state.reasoning_state == 1 {
            handler.text("\n</think>\n\n")?;
            state.reasoning_state = 0;
        }
        handler.text(text)?;
    } else if let Some(text) = data["choices"][0]["delta"]["reasoning_content"]
        .as_str()
        .or_else(|| data["choices"][0]["delta"]["reasoning"].as_str())
        .filter(|v| !v.is_empty())
    {
        if state.reasoning_state == 0 {
            handler.text("<think>\n")?;
            state.reasoning_state = 1;
        }
        handler.text(text)?;
    }
    for tool_call in data["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let Some(function) = tool_call["function"].as_object() else {
            continue;
        };
        let index = tool_call["index"]
            .as_u64()
            .or(state.current_index)
            .unwrap_or_default();
        let id = tool_call["id"].as_str().filter(|v| !v.is_empty());
        if state.reasoning_state == 1 {
            handler.text("\n</think>\n\n")?;
            state.reasoning_state = 0;
        }
        // Keep arrival order, including providers that reuse an index with a new id.
        let position = state
            .tool_calls
            .iter()
            .rposition(|call| call.index == index)
            .filter(|&position| {
                let call = &state.tool_calls[position];
                id.is_none()
                    || call.function_name.is_empty()
                    || id == Some(call.function_id.as_str())
            })
            .unwrap_or_else(|| {
                state.tool_calls.push(OpenAiStreamToolCall {
                    index,
                    ..Default::default()
                });
                state.tool_calls.len() - 1
            });
        state.current_index = Some(index);
        let call = &mut state.tool_calls[position];
        if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
            if name.starts_with(&call.function_name) {
                call.function_name = name.to_string();
            } else {
                call.function_name.push_str(name);
            }
        }
        if let Some(arguments) = function.get("arguments").and_then(|v| v.as_str()) {
            call.function_arguments.push_str(arguments);
        }
        if let Some(id) = id {
            call.function_id = id.to_string();
        }
    }
    Ok(false)
}

fn flush_openai_stream_tool_call(
    state: &mut OpenAiStreamToolCall,
    handler: &mut SseHandler,
) -> Result<()> {
    if state.function_name.is_empty() {
        return Ok(());
    }
    if state.function_arguments.is_empty() {
        state.function_arguments = String::from("{}");
    }
    let function_name = &state.function_name;
    let function_arguments = &state.function_arguments;
    let arguments: Value = function_arguments.parse().with_context(|| {
        format!("Tool call '{function_name}' have non-JSON arguments '{function_arguments}'")
    })?;
    handler.tool_call(ToolCall::new(
        state.function_name.clone(),
        arguments,
        normalize_function_id(&state.function_id),
    ))
}

pub async fn openai_embeddings(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<EmbeddingsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }
    let res_body: EmbeddingsResBody =
        serde_json::from_value(data).context("Invalid embeddings data")?;
    let output = res_body.data.into_iter().map(|v| v.embedding).collect();
    Ok(output)
}

#[derive(Deserialize)]
struct EmbeddingsResBody {
    data: Vec<EmbeddingsResBodyEmbedding>,
}

#[derive(Deserialize)]
struct EmbeddingsResBodyEmbedding {
    embedding: Vec<f32>,
}

pub fn openai_build_chat_completions_body(data: ChatCompletionsData, model: &Model) -> Value {
    let ChatCompletionsData {
        messages,
        temperature,
        top_p,
        functions,
        stream,
    } = data;

    let messages_len = messages.len();
    let messages: Vec<Value> = messages
        .into_iter()
        .enumerate()
        .flat_map(|(i, message)| {
            let Message { role, content } = message;
            match content {
                MessageContent::ToolCalls(MessageContentToolCalls {
                    tool_results,
                    text: _,
                    sequence,
                }) => {
                    if !sequence {
                        let tool_calls: Vec<_> = tool_results
                            .iter()
                            .map(|tool_result| {
                                json!({
                                    "id": tool_result.call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_result.call.name,
                                        "arguments": tool_result.call.arguments.to_string(),
                                    },
                                })
                            })
                            .collect();
                        let mut messages = vec![
                            json!({ "role": MessageRole::Assistant, "tool_calls": tool_calls }),
                        ];
                        for tool_result in tool_results {
                            messages.push(json!({
                                "role": "tool",
                                "content": tool_result.output.to_string(),
                                "tool_call_id": tool_result.call.id,
                            }));
                        }
                        messages
                    } else {
                        tool_results.into_iter().flat_map(|tool_result| {
                            vec![
                                json!({
                                    "role": MessageRole::Assistant,
                                    "tool_calls": [
                                        {
                                            "id": tool_result.call.id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_result.call.name,
                                                "arguments": tool_result.call.arguments.to_string(),
                                            },
                                        }
                                    ]
                                }),
                                json!({
                                    "role": "tool",
                                    "content": tool_result.output.to_string(),
                                    "tool_call_id": tool_result.call.id,
                                })
                            ]

                        }).collect()
                    }
                }
                MessageContent::Text(text) if role.is_assistant() && i != messages_len - 1 => {
                    vec![json!({ "role": role, "content": strip_think_tag(&text) }
                    )]
                }
                _ => vec![json!({ "role": role, "content": content })],
            }
        })
        .collect();

    let mut body = json!({
        "model": &model.real_name(),
        "messages": messages,
    });

    if let Some(v) = model.max_tokens_param() {
        if model
            .patch()
            .and_then(|v| v.get("body").and_then(|v| v.get("max_tokens")))
            == Some(&Value::Null)
        {
            body["max_completion_tokens"] = v.into();
        } else {
            body["max_tokens"] = v.into();
        }
    }
    if let Some(v) = temperature {
        body["temperature"] = v.into();
    }
    if let Some(v) = top_p {
        body["top_p"] = v.into();
    }
    if stream {
        body["stream"] = true.into();
    }
    if let Some(functions) = functions {
        body["tools"] = functions
            .iter()
            .map(|v| {
                json!({
                    "type": "function",
                    "function": v,
                })
            })
            .collect();
    }
    body
}

pub fn openai_build_embeddings_body(data: &EmbeddingsData, model: &Model) -> Value {
    json!({
        "input": data.texts,
        "model": model.real_name()
    })
}

pub fn openai_extract_chat_completions(data: &Value) -> Result<ChatCompletionsOutput> {
    let text = data["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();

    let reasoning = data["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .or_else(|| data["choices"][0]["message"]["reasoning"].as_str())
        .unwrap_or_default()
        .trim();

    let mut tool_calls = vec![];
    if let Some(calls) = data["choices"][0]["message"]["tool_calls"].as_array() {
        for call in calls {
            if let (Some(name), Some(arguments), Some(id)) = (
                call["function"]["name"].as_str(),
                call["function"]["arguments"].as_str(),
                call["id"].as_str(),
            ) {
                let arguments: Value = arguments.parse().with_context(|| {
                    format!("Tool call '{name}' have non-JSON arguments '{arguments}'")
                })?;
                tool_calls.push(ToolCall::new(
                    name.to_string(),
                    arguments,
                    Some(id.to_string()),
                ));
            }
        }
    };

    if text.is_empty() && tool_calls.is_empty() {
        bail!("Invalid response data: {data}");
    }
    let text = if !reasoning.is_empty() {
        format!("<think>\n{reasoning}\n</think>\n\n{text}")
    } else {
        text.to_string()
    };
    let output = ChatCompletionsOutput {
        text,
        tool_calls,
        id: data["id"].as_str().map(|v| v.to_string()),
        input_tokens: data["usage"]["prompt_tokens"].as_u64(),
        output_tokens: data["usage"]["completion_tokens"].as_u64(),
    };
    Ok(output)
}

fn normalize_function_id(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::create_abort_signal;
    use tokio::sync::mpsc::unbounded_channel;

    fn dispatched_tool_calls(chunks: &[&str]) -> Result<Vec<ToolCall>> {
        let (tx, _rx) = unbounded_channel();
        let mut handler = SseHandler::new(tx, create_abort_signal());
        let mut state = OpenAiChatStreamState::default();
        for data in chunks {
            let done = handle_openai_stream_message(
                &mut state,
                &mut handler,
                SseMmessage {
                    event: "message".into(),
                    data: (*data).to_string(),
                },
            )?;
            if done {
                break;
            }
        }
        Ok(handler.take().1)
    }

    #[test]
    fn glm5_omitted_id_continuation_accumulates_tool_args() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_95df1cc8dabc4b959bbbc431","type":"function","function":{"name":"web_search","arguments":""}}]}}],"model":"z-ai/glm-5-20260211","provider":"AtlasCloud"}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"max_results\": 10, \"query\": \"ESP32 command line development best practices\""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(
            calls[0].id.as_deref(),
            Some("call_95df1cc8dabc4b959bbbc431")
        );
        assert_eq!(
            calls[0].arguments,
            json!({
                "max_results": 10,
                "query": "ESP32 command line development best practices"
            })
        );
    }

    #[test]
    fn single_chunk_tool_call_dispatches_once() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_grok","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"hello\",\"max_results\":5}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].id.as_deref(), Some("call_grok"));
        assert_eq!(
            calls[0].arguments,
            json!({
                "query": "hello",
                "max_results": 5
            })
        );
    }

    #[test]
    fn multiple_tool_calls_in_one_chunk_are_all_dispatched() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"alpha","arguments":"{\"x\":1}"}},{"index":1,"id":"call_b","function":{"name":"beta","arguments":"{\"y\":2}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "alpha");
        assert_eq!(calls[0].id.as_deref(), Some("call_a"));
        assert_eq!(calls[0].arguments, json!({"x": 1}));
        assert_eq!(calls[1].name, "beta");
        assert_eq!(calls[1].id.as_deref(), Some("call_b"));
        assert_eq!(calls[1].arguments, json!({"y": 2}));
    }

    #[test]
    fn interleaved_tool_calls_accumulate_arguments_and_names_by_index() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"web_","arguments":"{\"query\":"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"weather","arguments":"{\"city\":\""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"search","arguments":"\"rust\"}"}},{"index":1,"function":{"arguments":"Oslo\"}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].id.as_deref(), Some("call_a"));
        assert_eq!(calls[0].arguments, json!({"query": "rust"}));
        assert_eq!(calls[1].name, "weather");
        assert_eq!(calls[1].id.as_deref(), Some("call_b"));
        assert_eq!(calls[1].arguments, json!({"city": "Oslo"}));
    }

    #[test]
    fn interleaved_calls_preserve_first_seen_order() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"beta","arguments":"{\"y\":"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"alpha","arguments":"{\"x\":1}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "beta");
        assert_eq!(calls[0].arguments, json!({"y": 2}));
        assert_eq!(calls[1].name, "alpha");
        assert_eq!(calls[1].arguments, json!({"x": 1}));
    }

    #[test]
    fn omitted_index_continues_the_current_call() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":3,"id":"call_a","function":{"name":"alpha","arguments":"{\"x\":"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"1}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_a"));
        assert_eq!(calls[0].arguments, json!({"x": 1}));
    }

    #[test]
    fn malformed_tool_arguments_keep_the_contextual_error() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"alpha","arguments":"{\"x\":"}}]}}]}"#,
            "[DONE]",
        ];
        let error = dispatched_tool_calls(&chunks).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Tool call 'alpha' have non-JSON arguments '{\"x\":'"
        );
    }

    #[tokio::test]
    async fn streamed_tool_calls_reach_the_handler_through_sse() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::time::Duration;

        let chunks = [
            r#"{"choices":[{"delta":{"reasoning_content":"Checking"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"alpha","arguments":"{\"x\":"}},{"index":1,"id":"call_b","function":{"name":"beta","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}"#,
            "[DONE]",
        ];
        let body = chunks
            .iter()
            .map(|data| format!("data: {data}\n\n"))
            .collect::<String>();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let request = client
            .get(format!("http://{address}/chat/completions"))
            .timeout(Duration::from_secs(5));
        let (tx, _rx) = unbounded_channel();
        let mut handler = SseHandler::new(tx, create_abort_signal());
        let result =
            openai_chat_completions_streaming(request, &mut handler, &Model::default()).await;
        server.join().unwrap();
        result.unwrap();
        let (text, calls) = handler.take();
        assert_eq!(text, "<think>\nChecking\n</think>\n\n");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "alpha");
        assert_eq!(calls[0].id.as_deref(), Some("call_a"));
        assert_eq!(calls[0].arguments, json!({"x": 1}));
        assert_eq!(calls[1].name, "beta");
        assert_eq!(calls[1].id.as_deref(), Some("call_b"));
        assert_eq!(calls[1].arguments, json!({}));
    }

    #[test]
    fn new_id_starts_new_tool_call() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_verylongidentifier","type":"function","function":{"name":"alpha","arguments":"{\"x\":1}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_z","type":"function","function":{"name":"beta","arguments":"{\"y\":2}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "alpha");
        assert_eq!(calls[0].id.as_deref(), Some("call_verylongidentifier"));
        assert_eq!(calls[0].arguments, json!({"x": 1}));
        assert_eq!(calls[1].name, "beta");
        assert_eq!(calls[1].id.as_deref(), Some("call_z"));
        assert_eq!(calls[1].arguments, json!({"y": 2}));
    }

    #[test]
    fn new_id_after_an_unidentified_call_at_the_same_index_starts_a_call() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"alpha","arguments":"{\"x\":1}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_b","function":{"name":"beta","arguments":"{\"y\":2}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "alpha");
        assert_eq!(calls[0].id, None);
        assert_eq!(calls[0].arguments, json!({"x": 1}));
        assert_eq!(calls[1].name, "beta");
        assert_eq!(calls[1].id.as_deref(), Some("call_b"));
        assert_eq!(calls[1].arguments, json!({"y": 2}));
    }

    #[test]
    fn new_index_starts_new_tool_call() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_aaa","type":"function","function":{"name":"alpha","arguments":"{\"x\":1}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_bbb","type":"function","function":{"name":"beta","arguments":"{\"y\":2}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "alpha");
        assert_eq!(calls[0].id.as_deref(), Some("call_aaa"));
        assert_eq!(calls[0].arguments, json!({"x": 1}));
        assert_eq!(calls[1].name, "beta");
        assert_eq!(calls[1].id.as_deref(), Some("call_bbb"));
        assert_eq!(calls[1].arguments, json!({"y": 2}));
    }

    #[test]
    fn new_index_without_id_starts_new_tool_call() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_aaa","type":"function","function":{"name":"alpha","arguments":"{\"x\":1}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"name":"beta","arguments":"{\"y\":2}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "alpha");
        assert_eq!(calls[0].id.as_deref(), Some("call_aaa"));
        assert_eq!(calls[0].arguments, json!({"x": 1}));
        assert_eq!(calls[1].name, "beta");
        assert_eq!(calls[1].id, None);
        assert_eq!(calls[1].arguments, json!({"y": 2}));
    }

    #[test]
    fn new_id_after_omitted_id_index_change_flushes_in_progress_call() {
        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_aaa","type":"function","function":{"name":"alpha","arguments":"{\"x\":1}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"name":"beta","arguments":"{\"y\":2}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":2,"id":"call_ccc","type":"function","function":{"name":"gamma","arguments":"{\"z\":3}"}}]}}]}"#,
            "[DONE]",
        ];
        let calls = dispatched_tool_calls(&chunks).unwrap();
        assert!(
            calls.len() >= 2,
            "expected distinct tool calls, got {calls:?}"
        );
        assert!(
            calls.iter().all(|call| call.arguments.is_object()),
            "arguments must not concatenate across calls, got {calls:?}"
        );
        assert_eq!(calls[0].name, "alpha");
        assert_eq!(calls[0].arguments, json!({"x": 1}));
        let last = calls.last().unwrap();
        assert_eq!(last.name, "gamma");
        assert_eq!(last.id.as_deref(), Some("call_ccc"));
        assert_eq!(last.arguments, json!({"z": 3}));
        assert!(
            calls
                .iter()
                .any(|call| { call.name == "beta" && call.arguments == json!({"y": 2}) }),
            "beta must be flushed before gamma, got {calls:?}"
        );
    }
}
