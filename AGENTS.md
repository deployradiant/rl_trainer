# Agent Notes

See README.md for architecture.

## Defaults

- Prefer simple, clean code over abstraction.
- Do not preserve backwards compatibility unless explicitly requested.
- Prefer fast implementation and verification loops over long upfront plans.
- Prefer deterministic tests and seeded randomness.
- Normalize timestamps to epoch timestamps in UTC.
- Track source, symbol, interval, and fetch time for market data.
- Never commit API keys, exchange secrets, wallet keys, or credentials.

## Subagent Workflow

- Subagents must always work in their own git worktree, not in the main local
  checkout.
- Use subagents for isolated investigation, experiments, and code changes only
  when their worktree can absorb the mess. Merge or copy back only the specific
  changes that have been reviewed.
- Do not let subagents write into the shared checkout unless the user explicitly
  requests that exception for a read-only or non-mutating task.

## Code Quality Review

Use the `code view` skill as the guide for strict maintainability review.

- Push for simple, direct implementations.
- Remove complexity instead of rearranging it.
- Reject spaghetti conditionals, weak boundaries, and unearned abstractions.
- Prefer breaking changes over compatibility shims while the project is young.

## Git Workflow

Agents should submit PR and ask for reviews.
