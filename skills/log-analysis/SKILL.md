---
name: log-analysis
description: Use this skill when analyzing log files, finding errors, counting log levels, extracting patterns, identifying root causes, or generating log analysis reports
---

# Log Analysis Skill

Specialized workflow for analyzing application log files.

## When to Use This Skill

Use this skill when asked to:
- Analyze a log file for errors, warnings, or patterns
- Count entries by log level (INFO/WARN/ERROR/DEBUG)
- Extract and summarize error messages with root causes
- Find slow queries, security events, or anomalies
- Generate a log analysis report

## Analysis Workflow

1. **Read the log data** - it may be provided in your task description (auto-injected) or you may need to use `read_file`
2. **Parse entries** - identify timestamp format, log levels, component tags, and message structure
3. **Categorize findings**:
   - Count by log level
   - Group errors by component/category
   - Identify patterns (recurring errors, timing correlations)
   - Flag security-related events
   - Note performance issues (slow queries, high memory)
4. **Produce a structured report**

## Output Format

```
# Log Analysis Report: <filename>

## Overview
- Time period: <start> to <end>
- Total entries: <count>
- Breakdown: <INFO count> INFO, <WARN count> WARN, <ERROR count> ERROR

## Critical Errors
### 1. <Error Category>
**Timestamp:** <time>
**Details:** <error message>
**Root Cause:** <analysis>
**Impact:** <what was affected>

## Warnings & Anomalies
- <warning summaries>

## Recommendations
1. <actionable recommendation>
```

## Efficiency Rules

- **STRICT SCOPE**: Only grep/search for what the task EXPLICITLY asks. If the task says "count ERROR entries", grep only for ERROR. Do NOT also count WARN, INFO, CRITICAL unless asked.
- **STOP IMMEDIATELY**: Once you have the answer to the specific question, write your response. No extra tool calls.
- **CALL BUDGET**: Simple count task = 2 tool calls max (1 to find file + 1 to grep/count).
- **NO AUTO-EXPANSION**: Do NOT add unrequested sections like "Key Error Patterns" or "Recommendations" for simple count queries.
- If log contents are already provided in the task, do NOT use `read_file` - analyze directly.

## ExACT-Inspired: Explore & Backtrack

Only applies when a tool call FAILS or returns empty results:

| Situation | Backtrack Strategy |
|---|---|
| grep returns 0 matches | Try grep with lowercase pattern, then read_file and scan manually |
| File not found at given path | Use ls on the directory, then glob pattern='**/*.log' to discover it |
| Log format unrecognized | Read first 20 lines to inspect actual format, then re-parse |

Always state: "I expected X, but got Y. I will now try Z instead."

## Multi-Angle Strategy (for MODERATE/COMPLEX logs)

For large or complex logs, the orchestrator may assign two workers:
- **Worker A - Statistical Pass**: count by level, timestamp range, frequency spikes per hour
- **Worker B - Root Cause Pass**: extract stack traces, identify failing components, correlated errors

The synthesizer merges both. State your worker role in your output header.