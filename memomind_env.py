import hashlib
import json
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional, Literal

# --- 1. Distributed Semantic Cache (Updated for Shared State) ---
class DistributedSemanticCache:
    def __init__(self, embedding_model, similarity_threshold: float = 0.85):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.accepted_validation_threshold = 0.75 # Minimum validation score to use cache

    def _get_embedding_hash(self, embedding: np.ndarray) -> str:
        return hashlib.sha256(embedding.tobytes()).hexdigest()

    def add_entry(self, input_text: str, output: Any, validation_score: float, step_id: str, is_final_output: bool = False, shared_flags: Dict[str, Any] = None) -> None:
        embedding = self.embedding_model.get_embedding(input_text)
        embedding_hash = self._get_embedding_hash(embedding)
        self.cache[embedding_hash] = {
            "original_input_text": input_text,
            "embedding": embedding,
            "output": output,
            "validation_score": validation_score,
            "timestamp": time.time(),
            "step_id": step_id,
            "is_final_output": is_final_output,
            "shared_flags": shared_flags if shared_flags is not None else {}
        }
        print(f"Cache: Added entry for '{input_text[:20]}...' from '{step_id}' with validation {validation_score:.2f}")

    def retrieve_similar(self, query_text: str, step_id: str = None) -> Optional[Dict[str, Any]]:
        query_embedding = self.embedding_model.get_embedding(query_text)
        best_match = None
        highest_similarity = -1.0

        for entry_hash, entry_data in self.cache.items():
            # Optional: Filter by step_id if relevant (e.g., only want output from specific step)
            if step_id and entry_data["step_id"] != step_id:
                continue

            cached_embedding = entry_data["embedding"]
            similarity = np.dot(query_embedding, cached_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding))

            if similarity >= self.similarity_threshold and similarity > highest_similarity:
                if entry_data["validation_score"] >= self.accepted_validation_threshold:
                    highest_similarity = similarity
                    best_match = entry_data

        if best_match:
            print(f"Cache: Found similar entry for '{query_text[:20]}...' (Sim: {highest_similarity:.2f})")
            return best_match
        # print(f"Cache: No sufficient match for '{query_text[:20]}...'")
        return None
    
    def get_shared_flag(self, flag_name: str) -> Any:
        """Retrieves a specific shared flag from the most recent relevant cache entry."""
        # This is a simplification. In reality, you'd have a dedicated shared state mechanism
        # or a more robust query for flags. Here, we look for the most recent flag.
        latest_timestamp = -1
        flag_value = None
        for entry_data in self.cache.values():
            if entry_data["timestamp"] > latest_timestamp and flag_name in entry_data["shared_flags"]:
                latest_timestamp = entry_data["timestamp"]
                flag_value = entry_data["shared_flags"][flag_name]
        return flag_value

    def set_shared_flag(self, flag_name: str, value: Any, step_id: str, context_input: str) -> None:
        """Sets a shared flag by adding a new cache entry with just the flag, or updating an existing one."""
        # For simplicity, we'll add a minimal entry to signify a shared flag update.
        # In a real system, you might have a dedicated "flag" cache or a mechanism
        # to update existing relevant entries without re-executing.
        self.add_entry(context_input, f"Flag update: {flag_name}={value}", 1.0, step_id, shared_flags={flag_name: value})
        print(f"Cache: Set shared flag '{flag_name}' to '{value}' by step '{step_id}'")

# --- 2. Conceptual LLM and Tool Models (Unchanged) ---
class ConceptualLLM:
    def generate(self, prompt: str, task_type: str, introduce_failure: bool = False) -> Tuple[str, bool]:
        # Simulate LLM generation with varying complexity/cost
        time.sleep(0.5 + len(prompt) * 0.001) # Simulate latency
        cost = 0.01 + len(prompt) * 0.0001 # Simulate cost
        
        # Introduce controlled failure for testing
        if introduce_failure and random.random() < 0.3: # 30% chance of failure if requested
            print(f"LLM: !!! SIMULATED FAILURE for '{task_type}' on '{prompt[:20]}...'")
            return "ERROR: LLM generation failed.", False # Output, success_status
        
        print(f"LLM: Generating for '{prompt[:20]}...' on task '{task_type}' (Cost: ${cost:.4f})")
        return f"Generated output for '{prompt}' on task '{task_type}'", True

class EmbeddingModel:
    def get_embedding(self, text: str) -> np.ndarray:
        time.sleep(0.01)
        return np.random.rand(100) # Dummy 100-dim embedding

class QualityValidator:
    def validate(self, original_input: str, generated_output: str) -> float:
        time.sleep(0.05)
        # Simulate varying quality, higher if 'success' or 'good' in output, lower if 'ERROR'
        if "ERROR" in generated_output:
            return 0.1 # Very low score for failed outputs
        return np.random.uniform(0.6, 1.0) if "good" in generated_output.lower() else np.random.uniform(0.3, 0.7)

# --- 3. PAM-like Step Definitions ---
from enum import Enum, auto
import random

class StepType(Enum):
    REQUIRED = auto()
    REQUISITE = auto()
    OPTIONAL = auto()
    SUFFICIENT = auto()

class CommonStep:
    def __init__(self, step_id: str, step_type: StepType, description: str, llm_model: ConceptualLLM, failure_rate: float = 0.0):
        self.step_id = step_id
        self.step_type = step_type
        self.description = description
        self.llm_model = llm_model
        self.failure_rate = failure_rate # Configurable failure rate for this step

    def execute(self, input_data: str) -> Tuple[str, float, float, bool]: # output, cost, latency, success
        start_time = time.time()
        # Introduce failure based on step's failure_rate
        introduce_failure = random.random() < self.failure_rate
        
        output, success = self.llm_model.generate(input_data, self.step_id, introduce_failure=introduce_failure)
        
        latency = time.time() - start_time
        cost = len(input_data) * 0.0005 # Example cost metric
        
        if success:
            print(f"Step '{self.step_id}': Executed SUCCESSFULLY. Latency: {latency:.2f}s, Cost: ${cost:.4f}")
        else:
            print(f"Step '{self.step_id}': Executed with FAILURE. Latency: {latency:.2f}s, Cost: ${cost:.4f}")
        
        return output, cost, latency, success

# --- Agent (New) ---
class Agent:
    def __init__(self, agent_id: str, responsible_step_types: List[StepType], llm_model: ConceptualLLM):
        self.agent_id = agent_id
        self.responsible_step_types = responsible_step_types
        self.llm_model = llm_model
        print(f"Agent '{self.agent_id}' initialized, responsible for: {[s.name for s in responsible_step_types]}")

    def can_handle(self, step_type: StepType) -> bool:
        return step_type in self.responsible_step_types

    def choose_action(self, state: Dict[str, Any], learned_policy_rules: Dict[StepType, Dict[str, Any]]) -> str:
        # Agent's policy logic based on state and its learned rules
        step_type = state["step_type"]
        cache_status = state["cache_status"]
        cached_output_validation = state["cached_output_validation"]
        
        if cache_status == "hit" and cached_output_validation >= env.accepted_validation_threshold:
            return learned_policy_rules[step_type].get("cache_hit_valid", "EXECUTE_COMPUTATION")
        else:
            return learned_policy_rules[step_type].get("default", "EXECUTE_COMPUTATION")

# --- 4. Reward Model (Unchanged, but calculations adapt to success/failure) ---
class RewardModel:
    def __init__(self):
        self.W_Req_Success = 100.0
        self.W_Req_Failure = -500.0
        self.W_Reqd_Success = 80.0
        self.W_Reqd_Failure = -400.0

        self.W_Opt_Success = 20.0
        self.W_Opt_Skipped = 10.0
        self.W_Opt_Failure = -5.0

        self.W_Suff_Cache = 150.0
        self.W_Suff_Compute = 50.0
        self.W_Redundant_Execution = -30.0

        self.C_Latency = 5.0
        self.C_Cost = 10.0
        self.W_Quality = 50.0

        self.W_Task_Completion = 1000.0
        self.W_Task_Failure = -1000.0
        
        self.W_Flag_Utility = 5.0 # Reward for setting useful flags
        self.W_ShortCircuit = 25.0 # Reward for successfully short-circuiting

    def calculate_reward(self, step_type: StepType, outcome: Literal["success", "failure", "skipped", "cache_hit_valid", "compute_success", "invalid_action"], current_cost: float, current_latency: float, quality_score: Optional[float] = None, short_circuited: bool = False, task_status: str = "ongoing") -> float:
        reward = 0.0

        # 1. PAM-Driven Rewards/Penalties
        if step_type == StepType.REQUISITE:
            if outcome in ["compute_success", "cache_hit_valid"]: reward += self.W_Req_Success
            elif outcome == "failure": reward += self.W_Req_Failure
        elif step_type == StepType.REQUIRED:
            if outcome in ["compute_success", "cache_hit_valid"]: reward += self.W_Reqd_Success
            elif outcome == "failure": reward += self.W_Reqd_Failure
        elif step_type == StepType.OPTIONAL:
            if outcome in ["compute_success", "cache_hit_valid"]: reward += self.W_Opt_Success
            elif outcome == "skipped": reward += self.W_Opt_Skipped
            elif outcome == "failure": reward += self.W_Opt_Failure
        elif step_type == StepType.SUFFICIENT:
            if outcome == "cache_hit_valid": reward += self.W_Suff_Cache
            elif outcome == "compute_success": reward += self.W_Suff_Compute
            if short_circuited: reward += self.W_ShortCircuit # Additional bonus for short-circuiting

        # 2. Efficiency Penalties
        reward -= self.C_Latency * current_latency
        reward -= self.C_Cost * current_cost

        # 3. Quality Reward (if applicable)
        if quality_score is not None:
            reward += self.W_Quality * quality_score

        # 4. Overall Task Completion Reward (Terminal State)
        if task_status == "completed": reward += self.W_Task_Completion
        elif task_status == "failed": reward += self.W_Task_Failure

        # 5. Shared State Impact Rewards (Conceptual)
        # This is more complex to detect automatically. An explicit action to set a flag,
        # or the environment could track if a step was *benefitted* by a prior flag.
        # For simulation, we'll implicitly add it if a sufficient step short-circuits.
        # if short_circuited: reward += self.W_Flag_Utility # Already covered by W_ShortCircuit effectively.

        return reward

# --- 5. RL Environment (Updated for Multi-Agent & Failure Handling) ---
class MemoMindEnv:
    def __init__(self, common_steps: Dict[str, CommonStep], cache: DistributedSemanticCache, reward_model: RewardModel, validator: QualityValidator, agents: List[Agent]):
        self.common_steps = common_steps
        self.cache = cache
        self.reward_model = reward_model
        self.validator = validator
        self.agents = {agent.agent_id: agent for agent in agents} # Map agent_id to Agent object
        
        self.current_task_input = ""
        self.current_step_idx = 0
        self.step_sequence_order = list(common_steps.keys())
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.task_outputs: Dict[str, Any] = {} # To store outputs of each step
        self.global_task_status = "ongoing" # Track overall task status
        
        # Policy rules will be managed externally (by Orchestrator/Trainer)
        self.learned_policy_rules: Dict[StepType, Dict[str, Any]] = {}

    def reset(self, task_input: str) -> Dict[str, Any]:
        self.current_task_input = task_input
        self.current_step_idx = 0
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.task_outputs = {}
        self.global_task_status = "ongoing"
        print(f"\n--- Environment Reset: Task Input '{task_input[:30]}...' ---")
        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        if self.global_task_status != "ongoing": # If task already failed or completed
            return {"terminal": True, "task_status": self.global_task_status}

        if self.current_step_idx >= len(self.step_sequence_order):
            return {"terminal": True, "task_status": "completed"}

        current_step_id = self.step_sequence_order[self.current_step_idx]
        current_step_obj = self.common_steps[current_step_id]
        
        # Input for current step might depend on previous step's output
        # For simplicity, we'll just use the initial task input, but in reality
        # it would be a composite of prior relevant outputs.
        input_for_step = f"Analyze '{self.current_task_input[:50]}...' for '{current_step_id}'."
        if current_step_id == "FinalAbstractCompose":
             # This step would gather all previous relevant outputs
             input_for_step = f"Compose abstract from: Intro Summary: {self.task_outputs.get('SummarizeIntro', 'N/A')[:50]}..., Keywords: {self.task_outputs.get('ExtractKeywords', 'N/A')[:30]}..., Method: {self.task_outputs.get('IdentifyMethod', 'N/A')[:50]}..., Results: {self.task_outputs.get('SynthesizeResults', 'N/A')[:50]}..."

        cache_match = self.cache.retrieve_similar(input_for_step, step_id=current_step_id) # Try to retrieve for specific step_id
        cache_status = "miss"
        cache_similarity = 0.0
        cached_output_validation = 0.0
        if cache_match:
            cache_status = "hit"
            cache_similarity = np.dot(self.cache.embedding_model.get_embedding(input_for_step), cache_match["embedding"])
            cached_output_validation = cache_match["validation_score"]

        # Shared state flags that might influence current step
        # E.g., a flag set by 'IdentifyMethod' that "paper_is_review" might influence 'SynthesizeResults'
        paper_type_flag = self.cache.get_shared_flag("paper_type")

        return {
            "terminal": False,
            "current_step_id": current_step_id,
            "step_type": current_step_obj.step_type,
            "input_for_step": input_for_step,
            "cache_status": cache_status,
            "cache_similarity": cache_similarity,
            "cached_output_validation": cached_output_validation,
            "time_budget_remaining": 60 - self.total_latency,
            "cost_incurred_so_far": self.total_cost,
            "cached_output": cache_match["output"] if cache_match else None,
            "shared_flag_paper_type": paper_type_flag # Example of reading shared state
        }

    def step(self, agent_id: str, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        current_state = self._get_state()
        if current_state["terminal"]:
            return current_state, 0.0, True, {"task_status": self.global_task_status}

        current_step_obj = self.common_steps[current_state["current_step_id"]]
        
        # Ensure the chosen agent is responsible for this step type
        acting_agent = self.agents[agent_id]
        if not acting_agent.can_handle(current_step_obj.step_type):
            print(f"Agent '{agent_id}' tried to handle step type {current_step_obj.step_type.name} it's not responsible for. Penalty!")
            reward = self.reward_model.calculate_reward(current_step_obj.step_type, "invalid_action", 0, 0)
            self.current_step_idx += 1 # Move on, but penalize
            return self._get_state(), reward, False, {"task_status": self.global_task_status}


        step_output = None
        step_cost = 0.0
        step_latency = 0.0
        quality_score = None
        outcome = "failure" # Default outcome if not explicitly set to success
        
        reward = 0.0
        done = False
        info = {}
        
        is_short_circuited = False # Flag for sufficient steps

        if action == "CHECK_CACHE_AND_USE_IF_VALID":
            print(f"Agent '{agent_id}' Action: CHECK_CACHE_AND_USE_IF_VALID for '{current_step_obj.step_id}'")
            cached_entry = self.cache.retrieve_similar(current_state["input_for_step"], step_id=current_step_obj.step_id)
            
            if cached_entry and cached_entry["validation_score"] >= self.cache.accepted_validation_threshold:
                step_output = cached_entry["output"]
                step_latency = 0.01
                step_cost = 0.0001
                outcome = "cache_hit_valid"
                quality_score = cached_entry["validation_score"]
                print(f" --> Cache HIT & VALID. Using cached output.")
                
                # If Sufficient, short-circuit
                if current_step_obj.step_type == StepType.SUFFICIENT:
                    is_short_circuited = True
                    self.global_task_status = "completed"
                    done = True # Task is done if Sufficient step is fulfilled via cache
                    print(f" --> SUFFICIENT Step '{current_step_obj.step_id}' fulfilled via cache. Short-circuiting!")
            else:
                print(f" --> Cache MISS or INVALID. Falling back to EXECUTE_COMPUTATION for '{current_step_obj.step_id}'.")
                # Fallback to computation, handling potential failure
                output, cost, latency, success = current_step_obj.execute(current_state["input_for_step"])
                step_output = output
                step_cost = cost
                step_latency = latency
                
                if success:
                    quality_score = self.validator.validate(current_state["input_for_step"], step_output)
                    self.cache.add_entry(current_state["input_for_step"], step_output, quality_score, current_step_obj.step_id)
                    outcome = "compute_success" if quality_score >= self.cache.accepted_validation_threshold else "failure" # Mark as failure if quality is too low
                else:
                    outcome = "failure" # Execution itself failed
                    quality_score = 0.0 # No quality from failed step
                
                # If Sufficient, and computed successfully, short-circuit
                if current_step_obj.step_type == StepType.SUFFICIENT and outcome == "compute_success":
                    is_short_circuited = True
                    self.global_task_status = "completed"
                    done = True # Task is done if Sufficient step is fulfilled via computation
                    print(f" --> SUFFICIENT Step '{current_step_obj.step_id}' computed. Short-circuiting!")

        elif action == "EXECUTE_COMPUTATION":
            print(f"Agent '{agent_id}' Action: EXECUTE_COMPUTATION for '{current_step_obj.step_id}'")
            output, cost, latency, success = current_step_obj.execute(current_state["input_for_step"])
            step_output = output
            step_cost = cost
            step_latency = latency
            
            if success:
                quality_score = self.validator.validate(current_state["input_for_step"], step_output)
                self.cache.add_entry(current_state["input_for_step"], step_output, quality_score, current_step_obj.step_id)
                outcome = "compute_success" if quality_score >= self.cache.accepted_validation_threshold else "failure"
            else:
                outcome = "failure"
                quality_score = 0.0

            # If Sufficient, and computed successfully, short-circuit
            if current_step_obj.step_type == StepType.SUFFICIENT and outcome == "compute_success":
                is_short_circuited = True
                self.global_task_status = "completed"
                done = True # Task is done
                print(f" --> SUFFICIENT Step '{current_step_obj.step_id}' computed. Short-circuiting!")

        elif action == "SKIP_STEP":
            print(f"Agent '{agent_id}' Action: SKIP_STEP for '{current_step_obj.step_id}'")
            if current_step_obj.step_type == StepType.OPTIONAL:
                outcome = "skipped"
                step_cost = 0.0
                step_latency = 0.01
                quality_score = 0.0
                print(f" --> OPTIONAL Step '{current_step_obj.step_id}' successfully SKIPPED.")
            else:
                outcome = "failure" # Cannot skip critical steps, treated as failure to execute
                step_cost = 0.01 # Small cost for trying
                step_latency = 0.01
                quality_score = 0.0
                print(f" --> Attempted to SKIP non-optional step '{current_step_obj.step_id}'. Treated as FAILURE.")

        else: # Invalid action
            outcome = "invalid_action"
            reward = -100 # Penalize invalid actions
            print(f" --> Invalid action '{action}' chosen by agent '{agent_id}'.")
            
        self.total_cost += step_cost
        self.total_latency += step_latency
        if step_output:
            self.task_outputs[current_step_obj.step_id] = step_output # Store output for subsequent steps / final composition
            
        # Update shared state based on step outcome (example)
        if current_step_obj.step_id == "IdentifyMethod" and outcome == "compute_success":
            # Simulate setting a flag based on method identification outcome
            if "literature review" in step_output.lower():
                self.cache.set_shared_flag("paper_type", "review", "IdentifyMethod", current_state["input_for_step"])
            else:
                self.cache.set_shared_flag("paper_type", "empirical", "IdentifyMethod", current_state["input_for_step"])

        # Calculate reward for this specific step's outcome
        reward += self.reward_model.calculate_reward(
            step_type=current_step_obj.step_type,
            outcome=outcome,
            current_cost=step_cost,
            current_latency=step_latency,
            quality_score=quality_score,
            short_circuited=is_short_circuited
        )

        # Check for task termination conditions
        if outcome == "failure" and (current_step_obj.step_type == StepType.REQUISITE or current_step_obj.step_type == StepType.REQUIRED):
            print(f"!!! Task FAILED at '{current_step_obj.step_id}' due to {outcome}. REQUISITE/REQUIRED step failed. !!!")
            self.global_task_status = "failed"
            done = True
            reward += self.reward_model.calculate_reward(task_status="failed")
            info["task_status"] = "failed"
        elif done: # If a Sufficient step already marked as done
            reward += self.reward_model.calculate_reward(task_status="completed")
            info["task_status"] = "completed"
        else: # Move to next step if not terminated
            self.current_step_idx += 1
            if self.current_step_idx >= len(self.step_sequence_order):
                self.global_task_status = "completed"
                done = True
                reward += self.reward_model.calculate_reward(task_status="completed")
                info["task_status"] = "completed"
        
        next_state = self._get_state() # Get state for the next step (or terminal state)
        
        return next_state, reward, done, info

# --- 6. Conceptual RL Agent (Policy Network) - Unchanged for each agent ---
# Each agent will conceptually have its own "policy" or contribute to a shared one
class ConceptualRLAgent:
    def __init__(self, agent_id: str, action_space: List[str]):
        self.agent_id = agent_id
        self.action_space = action_space
        # Each agent might have its own policy, or contribute to a central one.
        # For simplicity in this multi-agent demo, we assume they all adhere to the
        # same 'learned_policy_rules' managed by the Orchestrator.
        print(f"RL Agent '{self.agent_id}': Initializing.")

    def choose_action(self, state: Dict[str, Any], learned_policy_rules: Dict[StepType, Dict[str, Any]]) -> str:
        # This is where the RL policy network would make a decision.
        if state["terminal"]:
            return "NO_ACTION"

        step_type = state["step_type"]
        cache_status = state["cache_status"]
        cached_output_validation = state["cached_output_validation"]
        
        # Policy rules are passed in; agent acts on them
        if cache_status == "hit" and cached_output_validation >= env.cache.accepted_validation_threshold:
            return learned_policy_rules[step_type].get("cache_hit_valid", "EXECUTE_COMPUTATION")
        else:
            return learned_policy_rules[step_type].get("default", "EXECUTE_COMPUTATION")

    def learn(self, experiences: List[Tuple]) -> None:
        # In a real multi-agent RL, this could be central learning from all agents' experiences,
        # or decentralized learning. For this conceptual example, we'll keep it as a placeholder.
        # print(f"RL Agent '{self.agent_id}': Learning from {len(experiences)} experiences...")
        pass

# --- 7. Rule Extractor (Conceptual) - Unchanged ---
class RuleExtractor:
    def extract_rules_from_policy(self, learned_policy_rules: Dict[StepType, Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("Rule Extractor: Extracting explainable rules from learned policy...")
        extracted_rules = []
        for step_type, decisions in learned_policy_rules.items():
            rule = {
                "step_type": step_type.name,
                "conditions": [],
                "actions": {}
            }
            if "cache_hit_valid" in decisions:
                rule["conditions"].append("IF cache_hit_and_valid")
                rule["actions"]["cache_hit"] = decisions["cache_hit_valid"]
            if "default" in decisions:
                rule["conditions"].append("ELSE (cache miss or invalid)")
                rule["actions"]["default"] = decisions["default"]
            extracted_rules.append(rule)
        return extracted_rules

    def save_rules_to_pam_file(self, rules: List[Dict[str, Any]], filename: str = "memomind_protocol_multi_agent.json"):
        with open(filename, 'w') as f:
            json.dump(rules, f, indent=4)
        print(f"Rule Extractor: Rules saved to {filename}")

# --- 8. Orchestrator (New) ---
class Orchestrator:
    def __init__(self, env: MemoMindEnv, agents: List[Agent], reward_model: RewardModel):
        self.env = env
        self.agents = agents
        self.reward_model = reward_model
        # The Orchestrator will manage the shared policy rules that agents use
        self.learned_policy_rules = {
            StepType.REQUISITE: {"cache_hit_valid": "CHECK_CACHE_AND_USE_IF_VALID", "default": "EXECUTE_COMPUTATION"},
            StepType.REQUIRED: {"cache_hit_valid": "CHECK_CACHE_AND_USE_IF_VALID", "default": "EXECUTE_COMPUTATION"},
            StepType.OPTIONAL: {"cache_hit_valid": "CHECK_CACHE_AND_USE_IF_VALID", "default": "SKIP_STEP"},
            StepType.SUFFICIENT: {"cache_hit_valid": "CHECK_CACHE_AND_USE_IF_VALID", "default": "EXECUTE_COMPUTATION"},
        }
        print("Orchestrator initialized with initial policy rules.")

    def run_episode(self, task_input: str, is_training: bool = True) -> Tuple[float, str, List[Tuple]]:
        state = self.env.reset(task_input)
        episode_reward = 0
        done = False
        episode_experiences = []

        while not done:
            current_step_id = state["current_step_id"]
            current_step_obj = self.env.common_steps[current_step_id]
            current_step_type = current_step_obj.step_type

            # Find an agent capable of handling this step type
            responsible_agent = None
            for agent in self.agents:
                if agent.can_handle(current_step_type):
                    responsible_agent = agent
                    break
            
            if responsible_agent is None:
                print(f"No agent found for step type {current_step_type.name}. Task may be uncompletable.")
                # Penalize, terminate, or try a fallback
                reward = self.reward_model.W_Task_Failure / 2 # Moderate penalty
                episode_reward += reward
                done = True
                self.env.global_task_status = "failed"
                info = {"task_status": "failed", "reason": "no_agent_for_step"}
                break # Exit the loop if no agent can handle the step

            # Agent chooses action based on shared policy rules
            action = responsible_agent.choose_action(state, self.learned_policy_rules)
            
            # Record state before action
            prev_state_for_exp = state.copy()

            next_state, reward, done, info = self.env.step(responsible_agent.agent_id, action)
            episode_reward += reward
            
            if is_training:
                # Store experience for RL training
                episode_experiences.append((prev_state_for_exp, action, reward, next_state, done, responsible_agent.agent_id))
            
            state = next_state
        
        return episode_reward, self.env.global_task_status, episode_experiences

    def train_policy(self, experiences: List[Tuple]) -> None:
        # This is where a central RL algorithm would update the self.learned_policy_rules
        # based on all agents' experiences. For demonstration, we'll make a slight adjustment.
        print(f"Orchestrator: Training policy from {len(experiences)} experiences...")
        
        # Simulate very basic "learning" by subtly adjusting the policy
        # In a real system, a robust RL algorithm (e.g., PPO on collected trajectories)
        # would update the underlying neural network weights, and then you'd extract rules.
        
        # Example: If Optional steps were skipped and task completed, reinforce skipping
        # Or if Requisite steps failed, reinforce 'EXECUTE_COMPUTATION'
        
        # This is a *highly simplified* "learning" for the demo
        if random.random() < 0.5: # 50% chance to adjust optional step preference
            if self.learned_policy_rules[StepType.OPTIONAL].get("default") == "SKIP_STEP":
                self.learned_policy_rules[StepType.OPTIONAL]["default"] = "CHECK_CACHE_AND_USE_IF_VALID" # Try cache more often
                print("Orchestrator: Adjusting Optional step policy: now trying cache more often.")
            else:
                self.learned_policy_rules[StepType.OPTIONAL]["default"] = "SKIP_STEP"
                print("Orchestrator: Adjusting Optional step policy: now preferring to skip.")
        
        # In a real setup, a proper RL algorithm would update weights, and those weights
        # would implicitly define the rules. RuleExtractor would then interpret those weights.
        
# --- Simulation/Orchestration ---
if __name__ == "__main__":
    # Initialize Core Components
    embedding_model = EmbeddingModel()
    semantic_cache = DistributedSemanticCache(embedding_model)
    llm_model = ConceptualLLM()
    validator = QualityValidator()
    reward_model = RewardModel()

    # Define common steps with specific failure rates
    # Requisite/Required steps often have lower failure rates in real-world stable systems,
    # but we'll set some for demonstration of failure handling.
    common_steps = {
        "SummarizeIntro": CommonStep("SummarizeIntro", StepType.REQUIRED, "Summarize introduction section", llm_model, failure_rate=0.05), # 5% chance of failure
        "ExtractKeywords": CommonStep("ExtractKeywords", StepType.OPTIONAL, "Extract key terms", llm_model, failure_rate=0.1), # 10% chance of failure
        "IdentifyMethod": CommonStep("IdentifyMethod", StepType.REQUISITE, "Identify methodology section", llm_model, failure_rate=0.15), # 15% chance of failure
        "SynthesizeResults": CommonStep("SynthesizeResults", StepType.REQUIRED, "Synthesize results and conclusions", llm_model, failure_rate=0.07),
        "CheckGrammar": CommonStep("CheckGrammar", StepType.OPTIONAL, "Perform grammar and fluency check", llm_model, failure_rate=0.05),
        "FinalAbstractCompose": CommonStep("FinalAbstractCompose", StepType.SUFFICIENT, "Compose final abstract from parts", llm_model, failure_rate=0.02)
    }

    # Define Multiple Agents, specializing in step types
    agent_summarizer = Agent("Agent_Summarizer", [StepType.REQUIRED], llm_model)
    agent_analyst = Agent("Agent_Analyst", [StepType.REQUISITE, StepType.SUFFICIENT], llm_model) # Handles critical and sufficient steps
    agent_reviewer = Agent("Agent_Reviewer", [StepType.OPTIONAL], llm_model)

    all_agents = [agent_summarizer, agent_analyst, agent_reviewer]
    
    env = MemoMindEnv(common_steps, semantic_cache, reward_model, validator, all_agents)
    orchestrator = Orchestrator(env, all_agents, reward_model)
    rule_extractor = RuleExtractor()

    # --- Phase 1: Initial Ensemble Simulation & Cache Population ---
    print("\n--- Phase 1: Initial Ensemble Simulation & Cache Population ---")
    tasks = [
        "A research paper on new quantum computing algorithms focusing on error correction.",
        "Another paper about decentralized finance (DeFi) scalability solutions using rollups.",
        "A document focusing on AI ethics in autonomous driving systems.",
        "A document on quantum computing's energy efficiency challenges and prospects.", # Similar to first
        "Paper on decentralized ledger technologies for supply chain traceability." # Similar to second
    ]

    for i, task_input in enumerate(tasks):
        print(f"\nRunning Ensemble Pass {i+1} for: {task_input[:30]}...")
        # For initial ensemble, we can use a heuristic (e.g., always compute to fill cache)
        # or a simple, non-learning agent policy.
        state = env.reset(task_input)
        
        while not state["terminal"]:
            current_step_id = state["current_step_id"]
            current_step_obj = env.common_steps[current_step_id]
            current_step_type = current_step_obj.step_type

            # Find responsible agent (simple lookup for ensemble)
            responsible_agent = None
            for agent in all_agents:
                if agent.can_handle(current_step_type):
                    responsible_agent = agent
                    break
            
            if responsible_agent is None:
                print(f"Ensemble: No agent for step {current_step_id}. Halting task.")
                break

            # Ensemble heuristic: prefer cache if good, otherwise compute. Optional steps usually compute to populate.
            action_to_take = "EXECUTE_COMPUTATION" 
            if state["cache_status"] == "hit" and state["cached_output_validation"] >= env.cache.accepted_validation_threshold:
                action_to_take = "CHECK_CACHE_AND_USE_IF_VALID"
            # Optional steps can sometimes be skipped even in ensemble to test robustness
            if current_step_type == StepType.OPTIONAL and random.random() < 0.2:
                action_to_take = "SKIP_STEP"

            next_state, reward, done, info = env.step(responsible_agent.agent_id, action_to_take)
            print(f" --> Ensemble Action: {action_to_take}. Reward: {reward:.2f}. Total Cost: ${env.total_cost:.4f}")
            state = next_state
            if done:
                print(f"Ensemble Task finished with status: {info.get('task_status', 'Unknown')}")
                break

    print("\n--- Cache Population Complete ---")
    print(f"Cache entries: {len(semantic_cache.cache)}")

    # --- Phase 2: RL Agent Training (Orchestrator managing agents) ---
    print("\n--- Phase 2: RL Agent Training ---")
    training_episodes = 7 # More episodes for agents to learn
    total_experiences = []

    for episode in range(training_episodes):
        print(f"\n--- RL Training Episode {episode + 1} ---")
        task_input = random.choice(tasks)
        episode_reward, task_status, experiences = orchestrator.run_episode(task_input, is_training=True)
        total_experiences.extend(experiences)
        print(f"Episode {episode + 1} finished. Total Reward: {episode_reward:.2f}. Task Status: {task_status}")
    
    orchestrator.train_policy(total_experiences) # Orchestrator updates its shared policy rules
    
    # --- Phase 3: Rule Extraction & Deployment ---
    print("\n--- Phase 3: Rule Extraction & Deployment ---")
    extracted_rules = rule_extractor.extract_rules_from_policy(orchestrator.learned_policy_rules)
    rule_extractor.save_rules_to_pam_file(extracted_rules)

    # --- Phase 4: Final Agent Demonstration (Rule-Governed) ---
    print("\n--- Phase 4: Final Agent Demonstration (Rule-Governed) ---")
    # In deployment, the system would load these rules and route to appropriate agents
    final_agent_rules_loaded = extracted_rules # Assume these are loaded from the file

    new_task_input_1 = "A brief report on the latest trends in explainable AI for medical diagnosis."
    new_task_input_2 = "An analysis of carbon capture technologies and their economic viability." # Task that might trigger failure

    for demo_task_input in [new_task_input_1, new_task_input_2]:
        print(f"\nRunning Final Agent for: {demo_task_input[:30]}...")

        state = env.reset(demo_task_input)
        final_agent_total_reward = 0
        final_agent_done = False

        while not final_agent_done:
            current_step_id = state["current_step_id"]
            current_step_type = state["step_type"]

            responsible_agent = None
            for agent in all_agents: # In deployment, Orchestrator routes based on step_type
                if agent.can_handle(current_step_type):
                    responsible_agent = agent
                    break
            
            if responsible_agent is None:
                print(f"Final Agent: No responsible agent found for step {current_step_id}. Task likely failed prematurely.")
                break # Break out of loop as task cannot proceed

            # The final agent's decision logic directly uses the extracted rules
            chosen_action = "EXECUTE_COMPUTATION" # Default if no specific rule matches
            for rule in final_agent_rules_loaded:
                if rule["step_type"] == current_step_type.name:
                    if "IF cache_hit_and_valid" in rule["conditions"] and \
                       state["cache_status"] == "hit" and \
                       state["cached_output_validation"] >= env.cache.accepted_validation_threshold:
                        chosen_action = rule["actions"]["cache_hit"]
                        break
                    elif "ELSE (cache miss or invalid)" in rule["conditions"]:
                        chosen_action = rule["actions"]["default"]
                        break
            
            next_state, reward, final_agent_done, info = env.step(responsible_agent.agent_id, chosen_action)
            print(f" --> Final Agent chose Action: {chosen_action}. Reward: {reward:.2f}. Total Cost: ${env.total_cost:.4f}")
            final_agent_total_reward += reward
            state = next_state
            
        print(f"\nFinal Agent Task finished. Total Reward: {final_agent_total_reward:.2f}. Task Status: {info.get('task_status')}")
        print(f"Final Agent Total Cost: ${env.total_cost:.4f}, Total Latency: {env.total_latency:.2f}s")
