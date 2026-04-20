---
name: Config-vs-logfile timestamp skew
description: When diagnosing from a log file, compare config mtime to log mtime; if config is newer the user may have edited it after the bad run and the current values do not describe what was actually run.
type: feedback
---

When a user reports a bad run and hands you a log file and config, FIRST thing to check:

    stat config.yaml output/run.log

If config mtime > log mtime, the user likely edited the config after seeing the
bad behavior. Your plumbing checks against the CURRENT config may all pass
while the actual run used different values. In that case:

- Look at the `.adaptation.h5` trace's numerics (exact step_size values).
- Reverse-engineer what cap was actually in effect:
  ss_max can be read directly from any h5 entry where the trace plateaus.
- Ask the user to confirm what they changed before declaring "no bug."

**Why:** it's fast and it prevents a whole investigation chasing the wrong code.
**How to apply:** first bash command of any log-driven diagnosis.
