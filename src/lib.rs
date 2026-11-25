//! Official Rust SDK for [bf16.ai](https://bf16.ai) — an uncensored
//! 1M-context LLM inference API with automatic prompt caching and strict
//! Zero Data Retention (ZDR).
//!
//! # Quickstart
//!
//! ```no_run
//! use bf16::{ChatCompletionRequest, Client, Message, MODEL_DEEPSEEK_V41_FLASH_FAST};
//!
//! fn main() -> Result<(), bf16::Error> {
//!     let api_key = std::env::var("BF16_API_KEY").expect("BF16_API_KEY must be set");
//!     let client = Client::new(api_key);
//!
//!     // Check live USD balance and tier via GET /v1/me
//!     let me = client.me()?;
//!     println!("Balance: ${:.4} ({})", me.balance.available_usd, me.tier.name);
//!
//!     // Run a 1M-context chat completion
//!     let resp = client.chat_completion(&ChatCompletionRequest {
//!         model: MODEL_DEEPSEEK_V41_FLASH_FAST.to_string(),
//!         messages: vec![Message::user("Give 3 tips for prompt-cache hit rates.")],
//!         max_tokens: Some(256),
//!         temperature: Some(0.2),
//!     })?;
//!     if let Some(choice) = resp.choices.first() {
//!         println!("{}", choice.message.content);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! - Website & Console (`$0.20` starter credit): <https://bf16.ai>
//! - API Reference: <https://bf16.ai/docs>
//! - Live Model Catalog: <https://bf16.ai/models>
//! - Open-Source Inference Grants (`$50` API credits): <https://bf16.ai/open-source>

use serde::{Deserialize, Serialize};
use std::fmt;

/// Default OpenAI-compatible gateway base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.bf16.ai/v1";

/// Native Anthropic Messages endpoint (without `/v1`).
pub const ANTHROPIC_BASE_URL: &str = "https://api.bf16.ai";

/// Flagship 1M-context uncensored coding & reasoning model.
pub const MODEL_GLM_53: &str = "zai-org/glm-5.3-uncensored";

/// Vision-capable 1M-context flash model.
pub const MODEL_GLM_53_FLASH: &str = "zai-org/glm-5.3-flash-uncensored";

/// Low-cost 1M-context flash + vision model.
pub const MODEL_DEEPSEEK_V41_FLASH: &str = "deepseek/deepseek-v4.1-flash-uncensored";

/// High-throughput 1M-context flash + vision model.
pub const MODEL_DEEPSEEK_V41_FLASH_FAST: &str = "deepseek/deepseek-v4.1-flash-uncensored-fast";

/// 2.8T-parameter 1M-context reasoning & vision flagship.
pub const MODEL_KIMI_K31_PRO: &str = "moonshotai/kimi-k3.1-pro-uncensored";

/// Automatic smart router alias.
pub const MODEL_AUTO: &str = "bf16/auto";

/// Error type returned by [`Client`] methods.
#[derive(Debug)]
pub enum Error {
    /// Missing API key for an authenticated endpoint.
    MissingApiKey,
    /// HTTP status error returned by the bf16.ai gateway.
    Api { status: u16, message: String },
    /// Network or transport failure.
    Transport(String),
    /// JSON serialization or deserialization error.
    Json(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::MissingApiKey => write!(f, "bf16: API key is required"),
            Error::Api { status, message } => write!(f, "bf16 api error ({status}): {message}"),
            Error::Transport(msg) => write!(f, "bf16 transport error: {msg}"),
            Error::Json(msg) => write!(f, "bf16 json error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".to_string(), content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".to_string(), content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".to_string(), content: content.into() }
    }
}

/// Request payload for `POST /v1/chat/completions`.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// Token usage returned by completions.
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

/// A single choice in [`ChatCompletionResponse`].
#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: Message,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Response returned by `POST /v1/chat/completions`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Model entry from `GET /v1/models`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub owned_by: Option<String>,
}

/// Response from `GET /v1/models`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelInfo>,
}

/// Account USD balance returned by `GET /v1/me`.
#[derive(Debug, Clone, Deserialize)]
pub struct BalanceInfo {
    #[serde(default)]
    pub available_usd: f64,
    #[serde(default)]
    pub reserved_usd: f64,
}

/// Account tier metadata returned by `GET /v1/me`.
#[derive(Debug, Clone, Deserialize)]
pub struct TierInfo {
    #[serde(default)]
    pub id: u32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub max_concurrent_requests: u32,
    #[serde(default)]
    pub requests_per_minute: u32,
}

/// Account info returned by `GET /v1/me`.
#[derive(Debug, Clone, Deserialize)]
pub struct AccountInfo {
    pub balance: BalanceInfo,
    pub tier: TierInfo,
}

/// Lightweight synchronous client for the bf16.ai API.
#[derive(Debug, Clone)]
pub struct Client {
    api_key: String,
    base_url: String,
}

impl Client {
    /// Creates a client using [`DEFAULT_BASE_URL`] (`https://api.bf16.ai/v1`).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into().trim().to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Overrides the gateway base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Fetches the public model catalog from `GET /v1/models` (no API key required).
    pub fn list_models(&self) -> Result<ModelsResponse, Error> {
        let mut req = ureq::get(&format!("{}/models", self.base_url))
            .set("User-Agent", "bf16-rs/0.1.0");
        if !self.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        let resp = map_ureq(req.call())?;
        resp.into_json::<ModelsResponse>()
            .map_err(|e| Error::Json(e.to_string()))
    }

    /// Fetches live USD balance and tier limits from `GET /v1/me`.
    pub fn me(&self) -> Result<AccountInfo, Error> {
        if self.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let resp = map_ureq(
            ureq::get(&format!("{}/me", self.base_url))
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .set("User-Agent", "bf16-rs/0.1.0")
                .call(),
        )?;
        resp.into_json::<AccountInfo>()
            .map_err(|e| Error::Json(e.to_string()))
    }

    /// Executes `POST /v1/chat/completions`.
    pub fn chat_completion(
        &self,
        payload: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        if self.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let resp = map_ureq(
            ureq::post(&format!("{}/chat/completions", self.base_url))
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .set("User-Agent", "bf16-rs/0.1.0")
                .send_json(payload),
        )?;
        resp.into_json::<ChatCompletionResponse>()
            .map_err(|e| Error::Json(e.to_string()))
    }
}

fn map_ureq(result: Result<ureq::Response, ureq::Error>) -> Result<ureq::Response, Error> {
    match result {
        Ok(resp) => Ok(resp),
        Err(ureq::Error::Status(status, resp)) => {
            let msg = resp.into_string().unwrap_or_default();
            Err(Error::Api { status, message: msg })
        }
        Err(ureq::Error::Transport(t)) => Err(Error::Transport(t.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_helpers_set_roles() {
        assert_eq!(Message::system("s").role, "system");
        assert_eq!(Message::user("u").role, "user");
        assert_eq!(Message::assistant("a").role, "assistant");
    }

    #[test]
    fn client_trims_base_url_and_requires_key() {
        let client = Client::new("").with_base_url("https://api.bf16.ai/v1/");
        assert_eq!(client.base_url, "https://api.bf16.ai/v1");
        assert!(matches!(client.me(), Err(Error::MissingApiKey)));
    }
}
