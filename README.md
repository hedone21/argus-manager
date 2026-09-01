# Argus Manager

**System resource manager service** for the **Argus** on-device LLM inference
framework, written in Rust.

The manager runs as a separate process from the inference engine. It monitors system
resources (memory / CPU / GPU / temperature / power) and drives a scriptable policy that
tells the engine how much KV cache to keep, so inference adapts at runtime under memory
and thermal pressure.

The manager decides **when** relief is needed and **how much**; the engine decides
**which** of its KV cache techniques delivers it. That is why the contract carries a
budget rather than a technique name: adding or removing a technique in the engine does
not change the protocol, and an existing runtime can be integrated by implementing three
messages.

This repository is the **manager**. It is one of three Argus repositories:

| Repo | Role |
|------|------|
| [`argus-engine`](https://github.com/hedone21/argus-engine) | LLM inference engine |
| [`argus-shared`](https://github.com/hedone21/argus-shared) | IPC protocol types (manager ↔ engine) |
| [`argus-manager`](https://github.com/hedone21/argus-manager) | System resource manager service (this repo) |

```mermaid
graph LR
    Manager["Manager<br/>(monitor + policy)"]
    Engine["Engine<br/>(LLM inference)"]
    Manager -- "Directive (kv.compress { budget }, restore_defaults, suspend, resume)" --> Engine
    Engine -- "Heartbeat (kv bytes, TBT, phase) / Response (per-command result)" --> Manager
```

- **Transport:** Unix Domain Socket / TCP / D-Bus, serde JSON.
- **Policy:** a Lua script (`mlua`). The manager normalizes signals and applies
  enter/exit hysteresis; the script turns that into a KV budget.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `dbus` | ✅ | D-Bus IPC transport (via `zbus`) |
| `lua` | ✅ | Lua-scripted policy engine (via `mlua`, vendored Lua 5.4) |

## Build & run

```bash
cargo build --release
cargo test --workspace

# Run the manager service. The transport must be bidirectional — every decision
# needs the engine's heartbeat — so unix: or tcp:, not the emit-only dbus arm.
./target/release/argus-manager --policy-script scripts/policy_default.lua \
    --transport unix:/tmp/argus_manager.sock

# Mock peers for integration testing without a real counterpart:
./target/release/mock_manager   # stands in for the manager (run when testing an engine)
./target/release/mock_engine    # stands in for the engine (run when testing the manager)
./target/release/sim_run        # policy simulation harness (requires the `lua` feature)
```

Example Lua policies live in [`scripts/`](scripts/) (`policy_default.lua`,
`policy_example.lua`); see [`policy_config.toml`](policy_config.toml) for configuration.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your
option. Unless you state otherwise, contributions are dual licensed as above.
