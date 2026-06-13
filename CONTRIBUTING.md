# Contributing to argus-manager

Thanks for your interest in contributing!

`argus-manager` is the resource manager service that drives adaptive inference in
[`argus-engine`](https://github.com/hedone21/argus-engine) via the protocol types in
[`argus-shared`](https://github.com/hedone21/argus-shared).

## Development

```bash
cargo build                       # default: dbus + lua
cargo test --workspace
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

The `lua` feature vendors Lua 5.4 (compiles C). Tests use `insta` snapshots — run
`cargo insta review` to accept intentional snapshot changes.

## Conventions

- **Module file style:** no `mod.rs` — a directory module's root is the sibling `foo.rs`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) —
  `type(scope): subject`, imperative mood.

## License of contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you shall be dual licensed under **MIT OR Apache-2.0**,
without any additional terms or conditions.
