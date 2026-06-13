# Argus Manager — Agent Guide

Guidance for AI coding agents (Claude Code, Codex, Cursor, ...) and contributors.

## What this is

`argus-manager` is the system resource manager service of the Argus framework. It runs
as a separate process from [`argus-engine`](https://github.com/hedone21/argus-engine),
monitors system resources, and drives a scriptable policy engine (PI controller + Lua)
that sends `EngineCommand`s to adapt inference under memory/thermal/power pressure.
The wire types live in [`argus-shared`](https://github.com/hedone21/argus-shared).

Module layout (`src/`): `monitor` (resource sensing), `policy` (decision engine),
`relief` (pressure-relief actions), `emitter`/`channel` (IPC), `lua` (policy scripting),
`sim` (simulation harness).

## Working agreement

- **Think before coding.** State assumptions; surface trade-offs; ask when ambiguous.
- **Simplest thing that works.** No speculative abstraction or unrequested config.
- **Surgical changes.** Touch only what the task requires; match the surrounding style.
- **Compatibility.** Message types come from `argus-shared`; treat changes to existing
  serialized fields as breaking.

## Build & test

```bash
cargo build                       # default features: dbus + lua
cargo test --workspace            # includes insta snapshots and the sim harness
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

## Conventions

- **Module file style = no `mod.rs`** (a directory module's root is the sibling `foo.rs`).
- **Commits:** Conventional Commits — `type(scope): subject`, imperative mood.
- **License:** contributions are dual licensed `MIT OR Apache-2.0`.
