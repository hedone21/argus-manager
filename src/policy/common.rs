//! Shared policy state: normalized per-axis pressure and the enter/exit trigger.
//!
//! The EWMA relief table and its update events lived here too, feeding a DPP argmax over
//! ten named actions. With one command on the wire there is nothing to rank, so they went
//! with the action vocabulary; what remains is the hysteresis the paper's `decide` uses.

pub mod state;
