import os
import json
import ast  
import re
from openai import OpenAI
from server.environment import CodeReviewEnv
from models import CodeReviewAction
from tasks.task_definition import TASKS
from utils.embeddings_util import cosine_similarity, safe_embedding

# =========================
# ENV VARIABLES
# =========================

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable is required")

if not API_BASE_URL:
    raise ValueError("API_BASE_URL not set")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=OPENAI_API_KEY
)

TASK_NAMES  = ["easy", "medium", "hard"]
MAX_STEPS = 5  # single-step env

def safe_parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # fallback: handle single quotes or Python-style dicts
            return ast.literal_eval(content)
        except:
            # ultimate fallback
            return {
                "issues": ["unknown issue"],
                "severity": "low",
                "suggestion": "",
                "reasoning": ""
            }
        
def call_llm(prompt):
    
    if client is None:
        # 🔥 fallback response (VERY IMPORTANT)
        return """
        {
            "issues": ["Fallback: unable to analyze code"],
            "severity": "low",
            "suggestion": "Ensure API key is set for better analysis",
            "reasoning": "LLM not available, returning safe fallback"
        }
        """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return None
    
# =========================
# Prompt builder
# =========================
def build_prompt(code: str, feedback: str):
    return f"""You are a professional Python code reviewer.

    Code:
    {code}

    {f"Previous feedback from grader (use this to improve): {feedback}" if feedback else ""}

    Instructions:
    - List specific bugs, security flaws, or major inefficiencies.
    - Format "issues" as a simple list of strings: ["issue 1", "issue 2"].
    - Do NOT repeat points the feedback already rejected.
    - Ensure the JSON is complete and properly closed.
    - DO NOT repeat issues already identified in previous steps.

    Return ONLY valid JSON:
    {{
    "issues": ["string", "string"],
    "severity": "low|medium|high",
    "suggestion": "How to fix the code",
    "reasoning": "Why these changes are necessary"
    }}
    """

# =========================
# Utilities
# =========================
def clean_predicted_issues(predicted_issues):
    """Handle dicts, stringified dicts, or raw strings"""
    cleaned = []
    for i in predicted_issues:
        if isinstance(i, dict):
            cleaned.append(i.get("issue") or i.get("description") or str(i))
        else:
            # handle stringified dicts like "{'issue': 'xyz', ...}"
            match = re.search(r"'issue':\s*'([^']+)'", i)
            if match:
                cleaned.append(match.group(1))
            else:
                cleaned.append(i)
    return cleaned

def contains_keyword(text, keywords):
    if not text:
        return False
    text = text.lower()
    return any(k.lower() in text for k in keywords)

'''#using sentence-transformers for better semantic matching (optional, can be slow)
from difflib import SequenceMatcher
def similar(a, b, threshold=0.5):
    """Return True if strings are sufficiently similar"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold'''

def similar(a, b, threshold=0.75):
    emb_a = safe_embedding(a)
    emb_b = safe_embedding(b)

    # 🔥 fallback when API not available
    if emb_a is None or emb_b is None:
        return a.lower() in b.lower() or b.lower() in a.lower()

    return cosine_similarity(emb_a, emb_b) >= threshold

# =========================
# Grader
# =========================
def grade(task, action: CodeReviewAction):
    if task is None:
        return 0.0

    expected = task.get("expected", {})
    issues = expected.get("issues", [])
    expected_severity = expected.get("severity", "low")
    fix_keywords = expected.get("fix_keywords", [])
    concepts = expected.get("concepts", [])

    predicted_issues = clean_predicted_issues(action.issues or [])
    score = 0.0

    # 1️⃣ Issue detection
    matched = sum(1 for exp in issues if any(similar(exp, p) for p in predicted_issues))
    if issues:
        score += 0.4 * (matched / len(issues))

    # 2️⃣ Severity check
    severity_map = {"low": 1, "medium": 2, "high": 3}
    pred_sev = severity_map.get((action.severity or "").lower(), 0)
    exp_sev = severity_map.get(expected_severity.lower(), 0)
    if pred_sev and exp_sev:
        if pred_sev == exp_sev:
            score += 0.2
        elif abs(pred_sev - exp_sev) == 1:
            score += 0.1

    # 3️⃣ Suggestion check
    if contains_keyword(action.suggestion, fix_keywords):
        score += 0.2

    # 4️⃣ Reasoning check
    if contains_keyword(action.reasoning, concepts):
        score += 0.2

    # 5️⃣ Hallucination penalty
    unrelated = [p for p in predicted_issues if not any(similar(p, e) for e in issues)]
    if unrelated:
        score -= 0.02 * len(unrelated)

    return max(0.0, min(score, 1.0))

# =========================
# Inference / Runner
# ========================

# =========================
# Runner / Inference (multi-step, OpenAI gpt-5.4 ready)
# =========================

SUCCESS_THRESHOLD = 0.35  # reward needed to consider task successful

def run_task(task_name: str):
    env = CodeReviewEnv()
    task = TASKS[task_name]

    obs = env.reset(seed=42, episode_id=task_name)

    obs.code = task["code"]
    obs.feedback = ""
    obs.score = 0.0
    obs.remaining_issues = task["expected"]["issues"].copy()

    env._state.task_id = task_name
    env._state.task = task
    env._state.code = task["code"]
    env._state.remaining_issues = task["expected"]["issues"].copy()

    feedback = ""
    rewards = []
    success = False

    # ✅ REQUIRED FORMAT FIX
    print(
        f"[START] task={task_name} env=code-review model={MODEL_NAME}",
        flush=True
    )

    try:
        for step in range(1, MAX_STEPS + 1):

            prompt_text = build_prompt(obs.code, feedback)
            response = call_llm(prompt_text)

            if response is None:
                parsed = {
                    "issues": ["Fallback: unable to analyze code"],
                    "severity": "low",
                    "suggestion": "Ensure API key is set for better analysis",
                    "reasoning": "LLM not available, using fallback"
                }
            else:
                content = response.strip()
                content = content.split("}{")[0] + "}" if "}{" in content else content
                parsed = safe_parse_json(content)

            action = CodeReviewAction(**parsed)

            obs, reward, done, info = env.step(action)

	        # 🔥 FIX 2: Override broken remaining/done logic
            unmatched = [i for i in task["expected"]["issues"] 
                        if not any(similar(i, p) for p in (action.issues or []))]

            obs.remaining_issues = unmatched
            done = len(unmatched) == 0  # Continue until ALL issues fixed

            rewards.append(reward)

            error = info.get("last_action_error", None) if isinstance(info, dict) else None

            print(
                f"[STEP] step={step} "
                f"action={json.dumps(parsed)} "
                f"reward={reward:.2f} "
                f"done={str(done).lower()} "
                f"error={error if error else 'null'}",
                flush=True
            )

            feedback = f"""
            	Score: {obs.score}
            	Feedback: {obs.feedback}
            	Remaining issues: {obs.remaining_issues}
                """ 

            # stop if no improvement
            if len(rewards) >= 2 and rewards[-1] <= rewards[-2]:
                print(f"INFO =  No improvement, stopping early")
                success = max(rewards) >= 0.6
                break

            if reward >= 0.8 or done:
                success = True
                break

    except Exception as e:
        print(
            f"[STEP] step={step} action=null reward=0.00 done=true error={str(e)}",
            flush=True
        )
        success = False

    finally:
        env.close()

        # ✅ REQUIRED FORMAT FIX
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])

        print(
            f"[END] success={str(success).lower()} "
            f"steps={len(rewards)} "
            f"rewards={rewards_str}",
            flush=True
        )

# =========================
# Entry point
# =========================

if __name__ == "__main__":
    try:
        for task in TASKS:
            run_task(task)
    except Exception as e:
        print(f"[FATAL ERROR] {str(e)}")
        # Always return safe output
        print(json.dumps({
            "action": {
                "issues": ["Fatal runtime error"],
                "severity": "low",
                "suggestions": ["Check logs and fix exception"]
            }
        }))