# Attack Interface v2 — Policy + Session + Runner + Trajectories (Spec)

Status: draft proposal (no implementation; no backwards compatibility required).

## Prose description

In v2, an “attack” is no longer “a thing that runs one task couple end-to-end and returns a dict of injections”.
Instead, the system is decomposed into three roles:

1. **Session (executor)**: runs the *target* agent/environment state machine for a single episode (one task couple).
   It advances until it reaches a well-defined **decision point** (typically when injectable vectors exist), then pauses.
2. **Policy (attacker)**: a small piece of code that decides what to inject given the current observation.
   Policies can be single-step (template attacks), multi-step with internal state (GOAT/PAIR/TAP-like), human-in-the-loop,
   or batched/vectorized (RL-style inference). The policy does **not** run the target agent.
3. **Runner (orchestrator)**: runs many sessions concurrently, collects decision points into batches,
   calls the policy once per batch (`act_batch`), feeds actions back to sessions, and records **trajectories**
   (observation/action/outcome transitions) as a first-class output.

This architecture makes the attack interface resemble an RL environment:
sessions emit observations, the attacker emits actions, and the runner can produce rollouts.
Rollouts can be consumed by external training scripts (e.g., RL-hammer / rl-injector-style workflows)
without embedding training code inside prompt-siren.

## Goals / non-goals

### Goals

- **Batch-native attacks**: support policies that act on many episodes at once (`act_batch`), enabling efficient attacker inference
  and rollouts suitable for RL training scripts.
- **Trajectory-first**: export trajectories (rollouts) as a first-class output; training happens outside the attack itself.
- **Low boilerplate**: new attacks should often be implementable as “one async function from observation to action”.
- **Support common paradigms**:
  - template attacks
  - GOAT / PAIR / TAP-like attacks (iterative refinement with rollback/probing)
  - human-powered attacks
  - precomputed/file-based attacks
- **Future-proof**: allow new decision-point kinds later without redesigning everything.

### Non-goals

- Backwards compatibility with the existing `AbstractAttack.attack(...) -> (EndState, attacks_dict)` interface.
- Implementing any new attacks now.
- Implementing RL training inside prompt-siren.

## Terminology

- **Episode**: one run of one `TaskCouple`.
- **Decision point**: a pause where the policy is asked for an action.
  - v2 MVP decision point: reaching an `InjectableModelRequestState` (injection vectors are available).
- **Injection point**: a specific `InjectionVectorID` available at a decision point.
- **Action**: policy output for a decision point (payload updates + optional control).
- **Trajectory**: sequence of `(observation, action, outcome)` transitions for an episode.

## Architecture overview

### Components

1. **`AttackSession`**
   - Owns current agent execution state (`ExecutionState`).
   - Maintains an *accumulated injection mapping* (`vector_id -> payload`) across steps.
   - Advances target execution until the next decision point or termination.
   - Provides rollback/probing utilities for iterative attacks.

2. **`AttackPolicy`**
   - Consumes `AttackObservation`, returns `AttackAction`.
   - Optional `act_batch` for batch decisions.
   - Optional lifecycle hooks for caching/warmup/logging.
   - Must not directly drive the target agent/environment.

3. **`AttackRunner`**
   - Runs many sessions concurrently with a concurrency cap.
   - Batches observations and calls `policy.act_batch`.
   - Records trajectories at configurable fidelity.
   - Returns episode terminal results plus optional trajectories.

### Primary principle

Separate “**how to run/rewind the target**” (session) from “**how to choose injections**” (policy).
Batching is handled by the runner; policies opt into batching by implementing `act_batch`.

## Module layout (suggested)

Create a new subsystem (names are suggestions):

- `src/prompt_siren/attack_v2/types.py`
- `src/prompt_siren/attack_v2/session.py`
- `src/prompt_siren/attack_v2/policy.py`
- `src/prompt_siren/attack_v2/runner.py`
- `src/prompt_siren/attack_v2/registry.py`
- `src/prompt_siren/attack_v2/trajectory.py`
- Built-in policies under `src/prompt_siren/attack_v2/policies/…`

## Data model (runtime types)

Use `@dataclass(frozen=True)` for runtime records, and `BaseModel` only for config/serialization.

### `InjectionPoint`

Represents one injectable vector currently available.

Fields:

- `vector_id: InjectionVectorID`
- `location_kind: Literal["user_prompt", "tool_return", "retry_prompt"]`
- `tool_name: str | None`
- `tool_call_id: str | None`
- `default_payload: InjectionAttack` (what happens if attacker does nothing)
- `current_payload: InjectionAttack | None` (if previously set in accumulated mapping)
- `content_preview: str | None` (optional, best-effort human-facing preview)

### `AttackObservation`

Input to attacker policy.

Fields:

- `episode_id: str` (unique stable per episode)
- `couple_id: str` (existing `TaskCouple.id`)
- `step_idx: int` (decision-step count)
- `fsm_step: int` (raw FSM transition count; useful for debugging/probing)
- `benign_task_id: str`
- `malicious_task_id: str`
- `benign_prompt: str | list[UserContent | InjectableUserContent]`
- `malicious_goal: str`
- `messages: Sequence[ModelMessage | InjectableModelRequest]` (view-controlled; see “Observation views”)
- `injection_points: list[InjectionPoint]`
- `state_kind: Literal["injectable_request"]` (v2 MVP; extensible)
- `extras: dict[str, Any]` (optional extension surface)

### `AttackAction`

Output from policy.

Fields:

- `updates: dict[InjectionVectorID, InjectionAttack]`
- `control: Literal["continue", "stop"] = "continue"`
- `info: dict[str, Any]` (optional: logits/scores/chosen target/etc.)

Semantics:

- `updates` are merged into the session’s accumulated mapping; later updates overwrite earlier ones.
- `stop` means “do not ask me again; run the target agent to completion with current mapping”.

### `AttackOutcome`

Per-step outcome produced by the session.

Fields:

- `kind: Literal["next_decision", "done"]`
- `delta_usage: RunUsage` (since last decision point; best-effort)
- `new_messages: Sequence[ModelMessage | InjectableModelRequest] | None` (optional; view-controlled)
- `last_model_response: ModelResponse | None` (optional; useful for probing workflows)
- `info: dict[str, Any]`

### `AttackDone`

Terminal episode result at the executor layer.

Fields:

- `finish_reason: FinishReason`
- `run_ctx: RunContext[...]` (or `EndState` if preferred)
- `final_attacks: dict[InjectionVectorID, InjectionAttack]`
- `total_usage: RunUsage`
- `error: BaseException | None`

### `AttackTrajectory`

Primary RL export unit.

Fields:

- `episode_id: str`
- `couple_id: str`
- `transitions: list[AttackTransition]` where each transition contains:
  - `obs: AttackObservation` (maybe truncated)
  - `action: AttackAction`
  - `outcome: AttackOutcome`
- `final: AttackDone`
- `metadata: dict[str, Any]`

## Policy interface spec (`AttackPolicy`)

### Canonical protocol

- `name: ClassVar[str]`
- `config: BaseModel` property
- `async def act(self, obs: AttackObservation, ctl: AttackControl) -> AttackAction`

Optional:

- `async def act_batch(self, batch: Sequence[AttackObservation], ctl: AttackBatchControl) -> Sequence[AttackAction]`
- lifecycle hooks:
  - `async def on_experiment_start(self, ctx: AttackExperimentContext) -> None`
  - `async def on_experiment_end(self) -> None`
  - `async def on_episode_end(self, trajectory: AttackTrajectory | None, done: AttackDone) -> None`

### Functional, low-boilerplate authoring

Provide an adapter helper (spec-only):

- `policy_from_fn(name, config_model, act_fn, act_batch_fn=None, hooks=None)`

So the simplest attack can be:

- a config model
- `async def act_fn(obs, ctl) -> AttackAction`
- registry registration

### Policy constraints

- Policies must treat `episode_id` as the stable key for per-episode state.
- Policies should not mutate observations.
- Concurrency behavior should be explicit:
  - either “policy is thread-safe / concurrent-safe”
  - or runner guarantees single-threaded calls into a policy instance

## Control surfaces (probing/rollback without exposing env_state)

Policies sometimes need “try this injection, see what the target does, then revise”.
To support this without making policies drive the target directly, define control objects:

### `AttackControl` (per-episode)

Methods:

- `async def probe(self, action: AttackAction, *, until: Literal["model_response","next_decision","done"], max_fsm_steps: int | None = None) -> ProbeResult`
  - Applies action temporarily, advances the target FSM until the condition, captures results, then rolls back to the exact pre-probe point.

Notes:

- `probe` must restore state correctly for both snapshottable and non-snapshottable environments.
- `probe` must not permanently mutate the session’s accumulated mapping unless explicitly requested (default: no).

### `AttackBatchControl` (batched probing)

Methods mirror `AttackControl` but accept/return aligned sequences:

- `async def probe_batch(self, actions: Sequence[AttackAction], *, until=..., max_fsm_steps=...) -> Sequence[ProbeResult]`

## Session interface spec (`AttackSession`)

### Responsibilities

- Own the target `ExecutionState` and accumulated injections for one episode.
- Advance until the next decision point or termination.
- Apply injections at injection time by passing accumulated mapping into the target agent.
- Provide rollback/probing used by `AttackControl`.

### State held by a session

- `episode_id: str`
- `couple: TaskCouple`
- `current_state: ExecutionState | None`
- `acc_attacks: dict[InjectionVectorID, InjectionAttack]`
- `fsm_step: int`
- `decision_step: int`

Plus references:

- `agent`, `environment`, `toolsets`, `usage_limits`, `instrument`, `system_prompt`, `message_history`

### Decision points (v2 MVP)

- A decision point occurs when `current_state` is `InjectableModelRequestState`.
- If `EndState` is reached without any injectable state, the episode ends without decisions.

### `reset(...) -> AttackObservation | AttackDone`

Behavior:

1. Create per-task `env_state` via `environment.create_task_context(couple)`.
2. Initialize FSM via `agent.create_initial_request_state(...)`.
3. Advance until:
   - `InjectableModelRequestState` (emit `AttackObservation`), or
   - `EndState` (emit `AttackDone`).

### `step(action) -> tuple[AttackOutcome, AttackObservation | AttackDone]`

Behavior:

1. Merge `action.updates` into `acc_attacks` (action wins).
2. If `action.control == "stop"`: advance FSM to `EndState` without yielding further decision points.
3. Else:
   - If at an injectable state, apply injections by calling `agent.next_state(... attacks=acc_attacks ...)`
     to move past the injection application step.
   - Then continue advancing (`next_state`) until the next injectable state or end.
4. Return `AttackOutcome` and next `AttackObservation` or `AttackDone`.

### Rollback/checkpoint requirement

`probe` requires reliable rollback. The mechanism should be independent of object identity.

Spec approach:

- Define checkpoints as a logical position, e.g. `AttackCheckpoint(fsm_step: int, decision_step: int)`.
- Rollback by repeatedly calling `agent.prev_state(... toolsets=...)` until reaching the checkpoint.
  This reuses existing restoration logic (`restore_state_context`) for non-snapshottable environments.

### `probe(...) -> ProbeResult`

Semantics:

- Take checkpoint.
- Temporarily apply `action` (without permanent mutation by default).
- Advance until:
  - `model_response`: first `ModelResponseState` after applying injections
  - `next_decision`: next `InjectableModelRequestState`
  - `done`: `EndState`
- Capture:
  - usage delta
  - optionally the captured response / decision observation
- Roll back to checkpoint and restore accumulated mapping.

## Runner interface spec (`AttackRunner`)

### Responsibilities

- Manage `env.create_batch_context(couples)` for the run.
- Create one `AttackSession` per couple.
- Drive sessions concurrently until completion.
- Batch decision points into `policy.act_batch` calls.
- Record trajectories at configurable fidelity.

### Batching semantics

Runner config (suggested):

- `batch_size: int`
- `max_wait_ms: int` (how long to wait to fill a batch)
- `batch_order: Literal["arrival", "stable_sorted"]`
- `max_decision_steps_per_episode: int` (safety)
- `fail_fast: bool`

Contract:

- `act_batch` receives `B` observations and must return `B` actions in the same order.
- If `act_batch` is not implemented, runner falls back to calling `act` per observation.

### Safety/termination

Runner must enforce:

- maximum decision steps per episode (prevents infinite attacker loops)
- optional timeouts per episode

On safety violation:

- either force `stop` and finish, or mark episode as error (configurable).

### Outputs

Runner returns:

- `results: list[AttackDone]` aligned with input couples
- `trajectories: list[AttackTrajectory] | None` depending on recording config

Optional (recommended for RL):

- a streaming rollout API yielding trajectories as episodes finish.

## Trajectory recording configuration

`AttackTrajectoryConfig` (conceptual):

- `enabled: bool`
- `include_messages: Literal["none","last_k","all"]`
- `last_k: int = 40`
- `include_payloads: bool` (potentially sensitive; allow disabling)
- `include_previews: bool`
- `schema_version: str = "attack_v2_v1"`
- optional redaction settings (tool output / secrets)

## Observation views (policy-facing fidelity controls)

Runner-level config controls what policies see:

- `messages_view: Literal["none","last_k","all"]`
- `last_k: int`
- `include_content_preview: bool`

Rationale:

- Some attacks only need goal + injection vectors (keep observations small and stable).
- Human-powered attacks benefit from previews and more context.
- RL policies often want fixed-size observations (e.g., last-k messages).

## Registry & configuration spec

### Registry

Add a new entry point group (suggested):

- `prompt_siren.attack_policies`

Factory signature:

- `create_policy(config: ConfigT, context: None = None) -> AttackPolicy`

### Experiment config shape (conceptual)

Replace `attack: {type, config}` with:

```yaml
attack:
  policy:
    type: <policy_name>
    config: {...}
  runner:
    batch_size: 16
    max_wait_ms: 25
    max_decision_steps_per_episode: 20
  trajectory:
    enabled: true
    include_messages: last_k
    last_k: 40
  observation_view:
    messages_view: last_k
    last_k: 40
    include_content_preview: false
```

## Mapping to common attack paradigms

### Template attacks

- `act(obs, ctl)` chooses one or more `vector_id`s from `obs.injection_points` and returns payload updates.
- No probing required.

### GOAT / PAIR / TAP-like attacks

- Policy maintains per-episode attacker state keyed by `episode_id`.
- Inside `act`, policy can iteratively propose candidates and call `ctl.probe(...)` to observe target responses,
  then return a final committed `AttackAction`.

### Human-powered attacks

- Policy renders observation (messages + injection points) and requests user input.
- Returns an `AttackAction` with chosen vector payload(s).

### RL attacker / rl-injector-style workflow

- Policy is inference-only (current weights).
- Runner produces trajectories (streaming or collected).
- External training script consumes trajectories and updates the attacker; prompt-siren never implements training.

## Notes on integration with current prompt-siren primitives

- v2 MVP decision points map naturally to the existing `InjectableModelRequestState` and `InjectableModelRequestPart`.
- Rollback/probe semantics should reuse `AbstractAgent.prev_state` + `restore_state_context` to handle snapshottable vs non-snapshottable environments.
- The session should treat the accumulated attacks mapping as the single source of truth, passing it into `agent.next_state(... attacks=...)`
  when transitioning out of an injectable state.
- Persistence can keep storing `final_attacks` (like today’s `generated_attacks`) and optionally store trajectories as separate artifacts
  (distinct from `AttackFile`, which is “precomputed attacks by couple id”).

