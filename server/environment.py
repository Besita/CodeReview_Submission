import uuid
from tasks.task_definition import TASKS
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
import random
from utils.embeddings_util import cosine_similarity, safe_embedding

class CodeReviewEnv:

    def __init__(self, max_steps: int = 5):
        super().__init__()
        self.max_steps = max_steps
        # Initializing local state for this instance
        self._state = None

      
    def state(self) -> CodeReviewState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state
    
    
    def _get_task(self):
        return TASKS.get(self._state.task_id)

    def reset(self, seed: int = None, episode_id: str = None)  -> CodeReviewObservation:
    
        if seed is not None:   #Stabilize scoring (reproducibility)
            random.seed(seed)
        else:
            random.seed(42)  # default reproducibility 

        if seed is not None:
            random.seed(seed)
            task_id = random.choice(["easy", "medium", "hard"])
        else:
            task_id = "easy"

        task = TASKS[task_id]
        #task = self._get_task()
        #print("DEBUG TASK:", task)

        self._state = CodeReviewState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            task=task,                   # ✅ assign full task dictionary
            code=task["code"],
            remaining_issues=task["expected"]["issues"].copy(),
            done=False
        )        

        return CodeReviewObservation(
            code=self._state.code,
            score=0.1,
            feedback="Start review",
            remaining_issues=self._state.remaining_issues   # ✅ ADD THIS
        )

    def step(self, action: CodeReviewAction):
        if self._state is None:
            raise ValueError("Call reset() first")

        # ✅ Ensure "Episode Finished" score is strictly between 0 and 1
        if self._state.done:
            return CodeReviewObservation(code=self._state.code, score=0.1, feedback="Finished"), 0.1, True, {}
    
        try:
            state = self._state
            task = self._get_task()
            state.step_count += 1

            gt_issues = task["expected"]["issues"]
            user_issues = action.issues or []

            # 1. Semantic Matching
            gt_embeddings = [safe_embedding(gt) for gt in gt_issues]
            user_embeddings = [safe_embedding(ui) for ui in user_issues]

            similarity_matrix = []
           
            for gt_emb in gt_embeddings:
                row = [cosine_similarity(gt_emb, ui_emb) if gt_emb and ui_emb else 0.0 for ui_emb in user_embeddings]
                similarity_matrix.append(row)

            matched_scores = []
            used_users = set()
            for row in similarity_matrix:
                best_score, best_j = 0.1, -1
                for j, score in enumerate(row):
                    if j not in used_users and score > best_score:
                        best_score, best_j = score, j
                
                if best_j != -1 and best_score > 0.3:
                    used_users.add(best_j)
                    matched_scores.append(best_score)
                else:
                    matched_scores.append(0.1)

            # 2. Update Progress State
            state.remaining_issues = [gt for gt, s in zip(gt_issues, matched_scores) if s < 0.3]
            total_issues = max(1, len(gt_issues))
            resolved_issues = len(gt_issues) - len(state.remaining_issues)

            # 3. Component Scores
            semantic_score = sum(matched_scores) / total_issues
            
            # 4. Final Reward Assembly (Progressive & Positive)
            final_score = 0.2  # Baseline for valid JSON/Attempt
            final_score += (0.3 * semantic_score)
            final_score += (0.3 * (resolved_issues / total_issues))
            
            if (action.severity or "").lower() == task["expected"]["severity"].lower():
                final_score += 0.1
            
            # Light penalty for extra noisy predictions
            if len(user_issues) > len(gt_issues):
                final_score -= 0.05 * (len(user_issues) - len(gt_issues))

            final_score = float(max(0.01, min(final_score, 0.99)))     

            if final_score <= 0.0 or final_score >= 1.0:
                print("[ERROR] INVALID SCORE:", final_score)

            # Termination Logic
            task_complete = len(state.remaining_issues) == 0
            time_limit_hit = state.step_count >= self.max_steps
            state.done = task_complete or time_limit_hit
            
            obs = CodeReviewObservation(
                code=state.code,
                score=final_score,
                feedback=f"Step {state.step_count}: Resolved {resolved_issues}/{len(gt_issues)}",
                done=state.done,
                remaining_issues=state.remaining_issues
            )

            return obs, final_score, state.done, {}

        except Exception as e:
            print(f"[STEP ERROR] {e}")
            return CodeReviewObservation(code=self._state.code, score=0.01), 0.01, True, {"error": str(e)}


    async def reset_async(self):
        return self.reset()

    #async def step_async(self, action):
    #    return self.step(action)

    async def step_async(self, action, **kwargs):
        # This is what the server calls
        result = self.step(action)
        
        # If step returns a tuple, only give the server the first element
        if isinstance(result, tuple):
            return result[0] 
        return result
    
    def close(self):
        pass

    
