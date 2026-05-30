## Real-time audio constraints
- Avoid heap allocation, blocking I/O, and locks on the real-time audio callback path.
- Do not perform logging directly inside the audio callback unless already proven safe.
- Prefer preallocated buffers and message passing to background workers.
- Any change to chunking, buffering, or latency-sensitive code should be reviewed for real-time safety.

## Comments for future coding agents

When modifying non-trivial code, leave comments that help future AI agents
understand intent, constraints, and safe modification boundaries.

Prefer comments that explain:
- Why this implementation exists, especially when a simpler alternative looks tempting.
- Invariants that must remain true.
- Performance, latency, allocation, threading, or real-time constraints.
- Compatibility requirements, file format assumptions, or protocol expectations.
- Which nearby modules, tests, or configuration must be reviewed together when changing this code.
- Known trade-offs or intentionally rejected alternatives, when that context prevents accidental "cleanup".

Do not add comments that merely restate obvious code behavior.
Avoid filler comments such as "increment counter" or "loop over items".

When a future agent might incorrectly refactor or simplify a section,
add a concise guardrail comment explaining what must not be changed casually.