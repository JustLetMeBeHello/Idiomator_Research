# Pre-registration: E4 5-seed TOST (Joint F1, E4 vs D/E)

Committed 2026-06-23, before seeds {1, 99} are run. Locks the test so results can't be tuned post-hoc.

- **Test:** TOST equivalence, E4 Joint F1 vs System D and vs System E (two one-sided tests)
- **Margin:** Δ = ±0.02 (Joint F1 points), matching N1's existing convention
- **Mode:** bootstrap over per-example gaps pooled across 5 seeds (42, 123, 7, 1, 99), not seed-level summary stats
- **Alpha:** 0.05
- **Seeds committed:** 42, 123, 7 (existing) + 1, 99 (new)
- **Decision rule:** if both TOST tests reject (p < 0.05) → report E4 as equivalent to D/E on Joint F1. If either fails to reject → report directional difference + CI, no equivalence claim (same honesty pattern as N1's RemBERT result).
