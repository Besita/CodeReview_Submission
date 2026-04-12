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
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
#API_KEY = os.getenv("HF_TOKEN")

if not API_KEY:
    raise ValueError("No API key found (OPENAI_API_KEY / HF_TOKEN)")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

TASK_NAMES  = ["easy", "medium", "hard"]
MAX_STEPS = 6  # single-step env

FOCUS_AREAS = [
    "Security & Vulnerabilities (SQLi, Injection, Auth)",
    "Logic & Edge Cases (Null pointers, Division by zero, Bounds)",
    "Performance & Scalability (Big O complexity, Memory leaks)",
    "Readability & Standards (Naming, Formatting, Documentation)",
    "Architecture & Future-Proofing (Hardcoding, Scalability, Coupling)"
]


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
        return """
        {
            "issues": ["LLM failed"],
            "severity": "low",
            "suggestion": "Retry or check API",
            "reasoning": "LLM call failed"
        }
        """
    
def safe_reward(x) -> float:
    try:
        x = float(x)
    except:
        x = 0.5

    # HARD SAFETY
    if x is None or x != x:
        x = 0.5

    # STRICT BOUND (IMPORTANT)
    x = max(0.05, min(x, 0.95))

    return round(x, 2)
    
# =========================
# Prompt builder
# =========================
def build_prompt(code: str, feedback: str, step_num: int, audit_trail: list = None):
    # Step 6 is the Summary/Master Fix phase
    if step_num == 6:
        return f"""You are a Lead Software Architect.
        
        TASK:
        I have performed a 5-step audit on the code below. Use the provided AUDIT LOGS to 
        generate one final, "Master Fix" version of the code that resolves every issue found.
        
        ORIGINAL CODE:
        {code}
        
        AUDIT LOGS (Steps 1-5):
        {json.dumps(audit_trail, indent=2)}
        
        Return ONLY valid JSON:
        {{
            "fixed_code": "ONLY provide the fixed_code and a 1-sentence summary"
            "fixed_code": "...",
            "summary_of_changes": "..."

z        }}
        """

    # Steps 1-5: The Layered Audit phase
    focus_area = FOCUS_AREAS[step_num - 1]
    return f"""You are a professional Python code reviewer.
    Step: {step_num}/5
    Current Focus: {focus_area}

    Code:
    {code}

    {f"Previous feedback from grader: {feedback}" if feedback else ""}

    Instructions:
    - Focus EXCLUSIVELY on issues related to: {focus_area}.
    - Do NOT repeat issues already mentioned in previous steps.
    - If no critical issues are found:
        - Return an empty "issues" list
        - BUT still provide:
            - a minor observation OR
            - a potential improvement OR
            - a preventive best practice
    - DO NOT return completely empty reasoning.

    Return ONLY valid JSON:
    {{
    "issues": ["string"],
    "severity": "low|medium|high",
    "suggestion": "How to fix the code",
    "reasoning": "Why these changes are necessary"
    }}
    """

def grade_master_fix(master_fix, audit_trail):
    fixed_code = master_fix.get("fixed_code", "").lower()
    summary = master_fix.get("summary_of_changes", "").lower()
    
    # 1. Base Score (Baseline for valid structure)
    score = 0.3 
    
    # 2. Safety Check: If LLM failed or provided empty code
    if len(fixed_code) < 30:
        return 0.15

    # 3. Dynamic Audit Verification
    # Distribute 0.5 points across the 5 audit layers
    points_per_layer = 0.5 / len(audit_trail) 
    
    for entry in audit_trail:
        # Check if the master fix mentions the focus area or its specific suggestions
        focus_keywords = entry.get("focus", "").lower().split()
        suggestion_text = entry.get("suggestion", "").lower()
        
        # Award points if the final fix refers back to the audit findings
        if any(word in summary for word in focus_keywords if len(word) > 3):
            score += points_per_layer * 0.5
        if any(word in fixed_code for word in suggestion_text.split()[:10] if len(word) > 4):
            score += points_per_layer * 0.5

    # 4. Final Best Practice Bonuses
    if "docstring" in summary or '"""' in fixed_code: score += 0.09
    if "raise" in fixed_code or "try:" in fixed_code: score += 0.09

    # 5. Clip to ensure it's never exactly 0.0 or 1.0
    
    score=max(0.01, min(score, 0.95))
    return score


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

def similar(a, b, threshold=0.9):
    emb_a = safe_embedding(a)
    emb_b = safe_embedding(b)

    if emb_a is None or emb_b is None:
        return 0.55 if (a.lower() in b.lower() or b.lower() in a.lower()) else 0.25

    sim = cosine_similarity(emb_a, emb_b)

    # soft scaling instead of hard 0.95/0.3
    return 0.2 + 0.7 * sim   # always in (0.2, 0.9)

def best_match_score(exp, preds):
    if not preds:
        return 0.2

    score = max(similar(exp, p) for p in preds)

    # FINAL HARD SAFETY
    score = float(score)
    return max(0.05, min(score, 0.95))

# =========================
# Grader
# =========================
def grade(task, action: CodeReviewAction, step_num: int, previous_issues: list):
    try:
        if task is None: return 0.5  # safety fallback
    
        # --- 1. Calculate Base Performance (Your existing logic) ---
        # (Checking issues, severity, keywords, and concepts)
        base_score = 0.4  # Start with a baseline for a valid JSON response
        
        expected = task.get("expected", {})
        predicted_issues = clean_predicted_issues(action.issues or [])
        reasoning_text = (action.reasoning or "").lower()
        
        is_fallback = any("unknown issue" in str(p).lower() for p in predicted_issues)

        # --- 2. Calculate Base Performance ---
        if is_fallback:
            base_score = 0.2  # Low score for failed parsing
        elif not predicted_issues and "secure" in reasoning_text:
            base_score = 0.5  # High score for correctly identifying a clean area
        else:
            base_score = 0.4  # Baseline for a successful parse
            
            # Matching logic...
            matched_count = sum(1 for exp in expected.get("issues", []) 
                                if any(similar(exp, p) for p in predicted_issues))
            if expected.get("issues"):
                base_score += 0.3 * (matched_count / len(expected.get("issues")))

        # --- 3. Cross-Step Redundancy Check ---
        # Only check for repeats if they actually found issues
        if predicted_issues and not is_fallback:
            is_repeat = any(any(similar(curr, prev) for prev in previous_issues) for curr in predicted_issues)
            if is_repeat:
                base_score = base_score*0.5

        # --- 4. Strict Linear Multiplier ---
        # Growth rate 0.15 makes the climb feel more natural across 5 steps
        growth_rate = 0.15 
        multiplier = 1.0 + (step_num - 1) * growth_rate
        
        final_reward = base_score * multiplier

        if final_reward is None:
            final_reward = 0.1

        return safe_reward(final_reward)

    except Exception:
            # 🔥 NEVER crash grader
            return 0.5


# =========================
# Inference / Runner
# ========================

SUCCESS_THRESHOLD = 0.4  # reward needed to consider task successful

def run_task(task_name: str):
    env = CodeReviewEnv()

    if env is None:
        return 0.5  # neutral safe score
    
    task = TASKS[task_name]
    obs = env.reset(seed=42, episode_id=task_name)

    # Initialize task state
    obs.code = task["code"]
    env._state.task = task
    env._state.code = task["code"]

    history_of_issues = []
    rewards = []
    feedback = "" 
    audit_trail = [] # Collect logs for the master fix

    print(f"[START] task={task_name} env=code-review model={MODEL_NAME}", flush=True)

    try:
        # Phase 1: 5-Step Layered Audit
        for step in range(1, 6):

            error_msg = "null"

            prompt_text = build_prompt(obs.code, feedback, step)
            response = call_llm(prompt_text)

            # Robust JSON cleaning
            content = response.strip() if response else ""
            if content.startswith("```"):
                content = re.sub(r"```(?:json)?\n?|```", "", content).strip()
            
            parsed = safe_parse_json(content)
            
            # Save findings BEFORE the index error can occur
            audit_trail.append({
                "focus": FOCUS_AREAS[step-1],
                "issues": parsed.get("issues", []),
                "suggestion": parsed.get("suggestion", "")
            })

            action = CodeReviewAction(**parsed)
            reward = grade(task, action, step_num=step, previous_issues=history_of_issues)
                        
            history_of_issues.extend(clean_predicted_issues(action.issues or []))
            obs, _, _, _ = env.step(action)
            
            feedback = action.reasoning 
            rewards.append(reward)

            if not rewards:
                rewards = [0.1]

            if reward <= 0.0 or reward >= 1.0:
                print("[ERROR] Invalid reward:", reward)

            print(f"[STEP] step={step} action={json.dumps(parsed)} reward={reward:.2f} done=false error={error_msg}", flush=True)

        # Phase 2: Final Summary / Master Fix (Step 6)
        print("Generating Master Fix...", flush=True)
        # Call build_prompt with step_num=6 and the audit_trail
        summary_prompt = build_prompt(obs.code, feedback, 6, audit_trail=audit_trail)
        final_response = call_llm(summary_prompt)
        
        # Clean and parse the master fix
        summary_content = re.sub(r"```(?:json)?\n?|```", "", final_response).strip()
        master_fix = safe_parse_json(summary_content)
        
        summary_reward = grade_master_fix(master_fix, audit_trail)
        
        print(f"[STEP] step=6 action={json.dumps(master_fix)} reward={summary_reward:.2f} done=true error={error_msg}", flush=True)
        rewards.append(summary_reward)

    except Exception as e:
        error_msg=(e)
        print(f"[FATAL ERROR] {e}")

        # fallback action (VERY IMPORTANT)
        parsed = {
            "issues": ["runtime error"],
            "severity": "low",
            "suggestion": "fix exception",
            "reasoning": error_msg
        }

        reward = 0.1
        done = False

    total = sum(rewards)
    score = total / (len(rewards) + 1e-6)
    score = max(0.01, min(score, 0.99))
    success = score >= SUCCESS_THRESHOLD
   
    #best_reward = max(rewards) if rewards else 0.15
    #success = best_reward >= SUCCESS_THRESHOLD

    # 🔥 ALWAYS close env BEFORE END
    try:
        env.close()
    except Exception as e:
        print(f"[CLOSE ERROR] {e}")

    if not rewards:
        rewards = [0.1]

    reward_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={len(rewards)} "
        f"score={score:.3f} rewards={reward_str}",
        flush=True
    )

    
# =========================
# Entry point
# =========================


if __name__ == "__main__":
    try:
        print("Hello from OpenEnv!")
        for task in TASKS:
            run_task(task)
    except Exception as e:
        print(f" {str(e)}")
        # Always return safe output
        print(json.dumps({
            "action": {
                "issues": ["Fatal runtime error"],
                "severity": "low",
                "suggestions": ["Check logs and fix exception"]
            }
        }))