# `bf16` — Official Rust SDK for [bf16.ai](https://bf16.ai)

[![Crates.io](https://img.shields.io/crates/v/bf16.svg)](https://crates.io/crates/bf16)
[![Documentation](https://docs.rs/bf16/badge.svg)](https://docs.rs/bf16)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Lightweight Rust client for **[bf16.ai](https://bf16.ai)** — uncensored 1M-context open-weight LLM inference API (DeepSeek V4.1 Flash, GLM-5.3, Kimi K3.1 Pro) with automatic prompt caching and strict Zero Data Retention (ZDR).

## Install

```toml
[dependencies]
bf16 = "0.1"
```

## Quickstart

```rust
use bf16::{ChatCompletionRequest, Client, Message, MODEL_DEEPSEEK_V41_FLASH_FAST};

fn main() -> Result<(), bf16::Error> {
    let api_key = std::env::var("BF16_API_KEY").expect("BF16_API_KEY must be set");
    let client = Client::new(api_key);

    // Check live balance & concurrency tier via GET /v1/me
    let me = client.me()?;
    println!("Balance: ${:.4} ({})", me.balance.available_usd, me.tier.name);

    // Run a 1M-context chat completion
    let resp = client.chat_completion(&ChatCompletionRequest {
        model: MODEL_DEEPSEEK_V41_FLASH_FAST.to_string(),
        messages: vec![Message::user("Give 3 practical tips for prompt caching.")],
        max_tokens: Some(256),
        temperature: Some(0.2),
    })?;
    println!("{}", resp.choices[0].message.content);
    Ok(())
}
```

## Links

- Website & Console (`$0.20` starter credit): [https://bf16.ai](https://bf16.ai)
- API Docs: [https://bf16.ai/docs](https://bf16.ai/docs)
- Live Model Catalog & Pricing: [https://bf16.ai/models](https://bf16.ai/models)
- Coding Agent Cost Calculator: [https://bf16.ai/tools/cost-calculator](https://bf16.ai/tools/cost-calculator)
- Open-Source Inference Grants (`$50` API credits): [https://bf16.ai/open-source](https://bf16.ai/open-source)
- Official CLI (`npx bf16ai claude`): [https://github.com/BF16-AI/bf16](https://github.com/BF16-AI/bf16)

## License

MIT © BF16 Team
