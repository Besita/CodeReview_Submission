---
title: CodeReview Submission
emoji: 🐨
colorFrom: red
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: Code Review RL environment
---

🚀 Code Review RL Agent – Multi-Step Reflexion System
📌 Overview

This project implements a multi-step Reinforcement Learning (RL)-inspired code review agent that iteratively analyzes code, identifies issues across different dimensions, and generates a final Master Fix.

Instead of performing a single-pass review, the system uses a layered audit strategy + reward-based evaluation, simulating how a human expert progressively refines their understanding of code.

🧠 Core Idea

Traditional code review:

One-shot → shallow → inconsistent

This system:

Multi-step → structured → reward-driven → self-improving

🏗️ Architecture
🔄 Workflow
Code → Step-wise Review (5 Phases) → Master Fix → Grading → Reward
🔍 Multi-Step Audit (RL Loop)

The agent performs 5 structured review steps, each focusing on a different dimension:

FOCUS_AREAS = [
    "Security & Vulnerabilities (SQLi, Injection, Auth)",
    "Logic & Edge Cases (Null pointers, Division by zero, Bounds)",
    "Performance & Scalability (Big O complexity, Memory leaks)",
    "Readability & Standards (Naming, Formatting, Documentation)",
    "Architecture & Future-Proofing (Hardcoding, Scalability, Coupling)"
]
🧩 Step Behavior
Step	Focus Area	Goal
1	Security	Detect vulnerabilities
2	Logic	Catch runtime errors & edge cases
3	Performance	Identify inefficiencies
4	Readability	Improve clarity & standards
5	Architecture	Suggest scalable design improvements
🧾 Step Output Format

Each step produces structured JSON:

{
  "issues": ["..."],
  "severity": "low|medium|high",
  "suggestion": "...",
  "reasoning": "..."
}
🧠 Final Step – Master Fix

After 5 audit steps, the agent synthesizes all findings:

Input:
Original code
Full audit trail
Output:
{
  "fixed_code": "...",
  "summary_of_changes": "..."
}
🧠 RL Design (Conceptual)

Although not full RL training, this system mimics RL components:

RL Component	Implementation
State	Code + history + feedback
Action	Code review output
Reward	Grader score
Policy	LLM prompt strategy
Episode	One full 6-step run
🧮 Reward System
🔹 Step-Level Grading

Each step is scored based on:

Matching expected issues
Severity correctness
Reasoning quality
Redundancy penalty
reward = base_score * step_multiplier
🔹 Key Features
📈 Progressive scaling → later steps weighted higher
🔁 Duplicate penalty → discourages repetition
🧠 Semantic matching using embeddings
🧾 Master Fix Grading

The final output is evaluated based on:

Coverage of all audit findings
Alignment with suggestions
Code quality improvements
Best practices (docstrings, error handling)
🔐 Safety Constraints

To pass validation:

Rewards must be strictly in (0, 1)
No runtime crashes
Must always produce valid JSON
Must print required logs:
[START]
[STEP]
[END]
⚙️ Key Components
1. inference.py

Main execution script:

Runs tasks
Handles LLM calls
Controls multi-step loop
Logs outputs
2. Prompt Builder

Creates dynamic prompts per step:

build_prompt(code, feedback, step_num)
3. Grader
Step Grader:
grade(task, action, step_num, previous_issues)
Master Fix Grader:
grade_master_fix(master_fix, audit_trail)
4. Environment
CodeReviewEnv()

Handles:

State transitions
Observations
Episode flow
🧠 Anti-Repetition Mechanism

To avoid redundant outputs:

Tracks history_of_issues
Uses semantic similarity:
cosine_similarity(emb_a, emb_b)
Penalizes repeated findings
🔄 Execution Flow
python -m inference
Output Example
[START] task=easy env=code-review model=gpt-4.1-mini
[STEP]
step=1 ...
step=2 ...
...
step=6 MASTER FIX ...
[END] success=true steps=6 rewards=0.20,0.46,0.58,0.64,0.60
⚠️ Error Handling

Robust fallback mechanisms:

Safe JSON parsing (safe_parse_json)
LLM failure fallback responses
Embedding fallback when API unavailable
🌐 Environment Variables
Variable	Description
API_BASE_URL	Required (LiteLLM proxy endpoint)
MODEL_NAME	Model used (default: gpt-4.1-mini)
OPENAI_API_KEY	Optional (fallback supported)
HF_TOKEN	Optional
🧪 Design Strengths

✔ Structured reasoning
✔ Multi-dimensional analysis
✔ Reward-guided improvements
✔ Validator-safe execution
✔ Robust to missing APIs

🚧 Limitations
Not true RL training (no policy updates)
Dependent on LLM quality
Fixed step sequence (not adaptive yet)
🚀 Future Improvements
🔥 Reflexion-style critic loop
🔥 Tree-based search instead of linear steps
🔥 Learned reward model
🔥 Adaptive step termination
🧠 Key Insight

This system is not just “prompting” —
it’s controlled reasoning with feedback loops, inspired by RL.

📌 Summary

This project demonstrates how to:

Structure LLM reasoning into steps
Apply RL concepts without training
Build a robust evaluation-driven agent
Optimize for real-world constraints (validators, failures)
🙌 Final Note

This is a competition-ready, production-safe code review system designed to balance:

🧠 Intelligence + ⚙️ Reliability + 🧪 Evaluability

Happy Building 🚀

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
