import uuid
from tasks.task_definition import TASKS
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
import random
from utils.embeddings_util import cosine_similarity, safe_embedding
import traceback

import logging

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

    def step(self, action: CodeReviewAction) :

        if self._state is None:
            raise ValueError("Call reset() first")

        if self._state.done:
            return (
                CodeReviewObservation(
                    code=self._state.code,
                    score=0.1,
                    feedback="Episode already finished"
                ),
                0.1,
                True,
                {}
            )


        try:

            state=self._state
            
            task = self._get_task()

            if not task:
                raise ValueError(f"Invalid task_id: {self._state.task_id}")

            state.step_count += 1

            gt_issues = task["expected"]["issues"]
            user_issues = action.issues or []

            # Precompute embeddings
            gt_embeddings = [safe_embedding(gt) for gt in gt_issues]
            user_embeddings = [safe_embedding(ui) for ui in user_issues]

            # Build similarity matrix
            similarity_matrix = []
            for gt_emb in gt_embeddings:
                row = []
                for ui_emb in user_embeddings:
                    row.append(cosine_similarity(gt_emb, ui_emb))
                similarity_matrix.append(row)

            # Greedy matching
            matched_scores = []
            used_users = set()

            for i, row in enumerate(similarity_matrix):
                best_score = 0
                best_j = -1

                for j, score in enumerate(row):
                    if j not in used_users and score > best_score:
                        best_score = score
                        best_j = j

                if best_j != -1:
                    used_users.add(best_j)
                    matched_scores.append(best_score)
                else:
                    matched_scores.append(0)


            # 🔥 ADD HERE
            state.remaining_issues = [
                gt for gt, score in zip(gt_issues, matched_scores) if score < 0.25
            ]

            # Final semantic score
            #semantic_score = sum(matched_scores) / len(gt_issues)
            filtered_scores = [s for s in matched_scores if s > 0.25]
            #semantic_score = sum(filtered_scores) / len(gt_issues) 
            semantic_score = sum(filtered_scores) / max(1, len(gt_issues)) if filtered_scores else 0
            coverage = len(filtered_scores) / max(1, len(gt_issues))
           

            # 🔥 Strict penalty for wrong issue detection
            if semantic_score < 0.2:
                semantic_score *= 0.8

            # ========================
            # 🔥 Step 5: Reward Shaping
            # ========================

            # 1. Issue score
            issue_score = 0.7 * semantic_score + 0.3 * coverage

            # 2. Fix quality score
            fix_keywords = task["expected"]["fix_keywords"]
            #suggestion = action.suggestion.lower()
            suggestion = (action.suggestion or "").lower()


            fix_hits = 0
            for kw in fix_keywords:
                if kw in suggestion:
                    fix_hits += 1

            fix_score = fix_hits / len(fix_keywords) if fix_keywords else 0

            # 3. Concept understanding (semantic)
            concepts = task["expected"]["concepts"]
            #reasoning = action.reasoning
            reasoning = (action.reasoning or "")

            concept_scores = []
            r_emb = safe_embedding(reasoning)

            for concept in concepts:
                c_emb = safe_embedding(concept)
                concept_scores.append(cosine_similarity(c_emb, r_emb))

            concept_score = max(concept_scores) if concept_scores else 0

            # 4. Severity match
            expected_severity = task["expected"]["severity"]
            user_severity = action.severity

            severity_score = 0.99 if user_severity == expected_severity else 0.6
            
            # ------------------------
            # 1. Base score
            # ------------------------
            final_score = (
                0.5 * issue_score +
                0.2 * fix_score +
                0.2 * concept_score +
                0.1 * severity_score
            )

            # ------------------------
            # 2. Progress bonus
            # ------------------------
            total_issues = len(gt_issues)
            resolved_issues = total_issues - len(state.remaining_issues)

            progress_bonus = resolved_issues / max(1, total_issues)
            final_score += 0.6 * progress_bonus

            # ------------------------
            # 3. Penalties (capped)
            # ------------------------
            extra_predictions = len(user_issues) - len(gt_issues)
            hallucination_penalty = 0.1 * max(0, extra_predictions)
            missing_penalty = 0.02 * matched_scores.count(0)

            total_penalty = hallucination_penalty + missing_penalty
            total_penalty = min(total_penalty, 0.5)

            final_score -= total_penalty

            # ------------------------
            # 4. Step penalty (small)
            # ------------------------
            step_penalty = 0.01 * state.step_count
            final_score -= step_penalty

            # ------------------------
            # 5. FINAL CLAMP (VERY IMPORTANT)
            # ------------------------
            final_score = max(0, min(final_score, 1))

            #termination logic
            task_complete = len(state.remaining_issues) < len(gt_issues) 
            time_limit_hit = state.step_count >= self.max_steps
            
            state.done = task_complete or time_limit_hit
           

            #feedback = f"Step {self._state.step_count}: Score {step_score:.2f}"
            feedback = (
                f"Step {self._state.step_count} | "
                f"Issue: {issue_score:.2f} | "
                f"Fix: {fix_score:.2f} | "
                f"Concept: {concept_score:.2f} | "
                f"Severity: {severity_score:.2f}"
            )

            if hallucination_penalty > 0:
                feedback += " | Hallucinated issues"
            if missing_penalty > 0:
                feedback += " | Missed critical"

            if time_limit_hit and not task_complete:
                feedback += " | ⚠️ TERMINATED: Maximum steps reached."

            # ✅ Persist feedback into state (multi-step memory)
            if not state.feedback:
                state.feedback = feedback
            else:
                state.feedback += " | " + feedback
            
            return (
            CodeReviewObservation(
                code=state.code,
                score=final_score,
                feedback=state.feedback,
                done=state.done   # ✅ also include this
            ),
            float(final_score),
            state.done,
            {
                "remaining_issues": state.remaining_issues,  # ✅ full list here
                "step_count": state.step_count
            }
        )
                    
        except Exception as e:
            print("❌ ERROR IN STEP FUNCTION:")
            traceback.print_exc() 
            # This allows the app to stay alive but tells you WHY it failed in the console
            raise e 

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

    
