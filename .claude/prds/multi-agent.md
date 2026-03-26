---
name: multi-agent
description: Multi-agent extension for Minigrid with simultaneous actions, per-agent rewards, and PettingZoo ParallelEnv interface
status: backlog
created: 2026-03-25T23:12:48Z
---

# PRD: multi-agent

## Executive Summary

Extend Minigrid with a multi-agent system (`minigrid.multigrid`) that supports simultaneous agent actions, per-agent rewards, partial observability, and the PettingZoo `ParallelEnv` interface. The system is designed for MARL research with both cooperative and competitive settings. The architecture is extensible toward future dialog/communication actions beyond physical grid actions.

## Problem Statement

Minigrid is a widely-used lightweight grid-world environment for RL research, but it only supports a single agent. Multi-agent reinforcement learning (MARL) researchers must either:

1. Use separate libraries (e.g., PettingZoo + custom envs) that lack Minigrid's simplicity and extensibility.
2. Hack single-agent Minigrid to simulate multiple agents, losing clean abstractions.
3. Use heavier frameworks (e.g., MarlGrid) that are poorly maintained or diverge from upstream Minigrid.

There is no clean, maintained, PettingZoo-compatible multi-agent extension that preserves Minigrid's design philosophy: lightweight, fast, readable, and easy to customize.

## User Stories

### US-1: MARL researcher creates a cooperative environment
**As a** MARL researcher, **I want to** create a multi-agent grid environment where agents must cooperate to reach a shared goal, **so that** I can study cooperative strategies.

**Acceptance criteria:**
- Can instantiate an environment with N agents (N specified at init)
- Each agent receives its own partial observation
- Environment returns per-agent rewards as a dict
- Environment follows PettingZoo `ParallelEnv` interface
- Agents block each other's movement (cannot occupy same cell)

### US-2: MARL researcher creates a competitive environment
**As a** MARL researcher, **I want to** create a competitive grid environment where agents race to a goal or collect resources, **so that** I can study competitive dynamics.

**Acceptance criteria:**
- Per-agent rewards can differ (positive for one, negative/zero for others)
- Agents can interact with the same objects (e.g., pick up the same key type)
- Terminated/truncated flags are per-agent (agents can finish at different times)

### US-3: Researcher switches between full and partial observability
**As a** researcher, **I want to** toggle between MDP (full grid observation) and POMDP (partial agent-centric view) settings, **so that** I can study the effect of observability on multi-agent coordination.

**Acceptance criteria:**
- Default is POMDP (each agent sees its own partial view, same as single-agent Minigrid)
- A configuration flag enables MDP mode (each agent receives the full grid state)
- Other agents appear as colored objects in both observation modes

### US-4: Researcher builds a custom multi-agent environment
**As a** researcher, **I want to** subclass a base multi-agent environment and implement `_gen_grid()`, **so that** I can create new scenarios with minimal boilerplate.

**Acceptance criteria:**
- Subclassing requires only implementing `_gen_grid()` (same pattern as single-agent Minigrid)
- Helper methods exist: `place_agent(agent_index)`, `place_obj()`, `put_obj()`
- Agents are configured via a simple parameter (e.g., `num_agents=3` or list of agent configs)

### US-5: Researcher integrates with standard MARL training libraries
**As a** researcher, **I want to** use this environment with PettingZoo-compatible training frameworks (e.g., RLlib, TorchRL, CleanRL), **so that** I can focus on algorithms rather than env wrappers.

**Acceptance criteria:**
- Environment passes PettingZoo `parallel_api_test`
- `possible_agents`, `agents`, `observation_space()`, `action_space()` all work correctly
- `reset()` returns `(observations, infos)` as dicts keyed by agent name
- `step(actions)` returns `(observations, rewards, terminations, truncations, infos)` as dicts

## Functional Requirements

### FR-1: Core Multi-Agent Environment Base Class
- `MultiGridEnv` extends PettingZoo `ParallelEnv`
- Manages a shared `Grid` (reuses existing Minigrid `Grid` class)
- Stores a list of `AgentState` objects, each with: position, direction, color, carrying, active/done flag
- Agent names follow pattern: `"agent_0"`, `"agent_1"`, ..., `"agent_{n-1}"`
- Number of agents is fixed at construction (no mid-episode changes)

### FR-2: Simultaneous Action Execution
- `step(actions: dict[str, int])` accepts one action per active agent
- All actions are resolved simultaneously per timestep
- Collision resolution: if two agents attempt to move to the same cell, neither moves
- Agent-object interactions (pickup, drop, toggle) are processed after movement
- Action space per agent: same 7 discrete actions as single-agent Minigrid (left, right, forward, pickup, drop, toggle, done)

### FR-3: Per-Agent Observations
- Each agent receives its own observation dict: `{"image": ndarray, "direction": int, "mission": str}`
- POMDP mode (default): agent sees a rotated partial view (default 7x7) centered on itself
- MDP mode (configurable): agent sees the full grid encoded as an ndarray
- Other agents appear in observations encoded as colored agent objects (type=10 in existing encoding, with color index distinguishing agents)

### FR-4: Per-Agent Rewards, Terminations, Truncations
- `step()` returns dicts keyed by agent name for: rewards, terminations, truncations, infos
- An agent that is terminated/truncated is removed from `agents` list (PettingZoo convention)
- Episode ends when all agents are terminated/truncated, or `max_steps` is reached
- Reward logic is defined by subclass (no enforced cooperative/competitive structure)

### FR-5: Rendering
- Full grid rendering shows all agents as colored triangles (each agent gets a distinct color)
- Agent colors: configurable, defaults to a predefined palette (red, blue, green, purple, yellow, grey)
- Rendering reuses existing `Grid.render()` and `render_tile()` infrastructure
- Support both "human" (pygame) and "rgb_array" render modes

### FR-6: Grid and Object Reuse
- Reuse existing `Grid`, `WorldObj`, and all object subclasses (Wall, Door, Key, Ball, Box, Goal, Lava) without modification
- Add an `AgentObj` world object subclass for rendering other agents in the grid
- Existing constants, encoding, and tile rendering are extended minimally

### FR-7: Example Environments
- `MultiGrid-Empty-v0`: N agents in an empty room, each with their own goal
- `MultiGrid-DoorKey-v0`: Cooperative — one agent must hold a door open while another passes through
- `MultiGrid-Adversarial-v0`: Competitive — agents race to reach a single goal first
- All registered as Gymnasium/PettingZoo environments

### FR-8: Observability Configuration
- Constructor parameter `full_obs: bool = False`
- When `True`, each agent's "image" observation is the full grid encoding
- When `False`, each agent gets a partial view (standard Minigrid behavior)

## Non-Functional Requirements

### NFR-1: Performance
- Environment step throughput should be within 2x of single-agent Minigrid for 2-4 agents
- No per-step memory allocations beyond observation arrays

### NFR-2: Code Quality
- Follows existing Minigrid code style and conventions
- Type hints on all public methods
- Docstrings on all public classes and methods

### NFR-3: Compatibility
- Python 3.9+ (matches Minigrid)
- PettingZoo >= 1.24.0
- No changes to existing single-agent Minigrid code (additive only)
- Existing tests continue to pass

### NFR-4: Testability
- Unit tests for core multi-agent step logic (movement, collision, pickup/drop)
- PettingZoo API conformance test passes
- Example environments have smoke tests

## Success Criteria

1. `MultiGridEnv` passes PettingZoo `parallel_api_test` with 2, 3, and 4 agents
2. All 3 example environments are functional and renderable
3. A simple MARL training loop (e.g., independent PPO) can train on `MultiGrid-Empty-v0`
4. Existing single-agent Minigrid tests pass with zero changes
5. Step throughput for 2-agent `MultiGrid-Empty-v0` is within 2x of single-agent `MiniGrid-Empty-8x8-v1`

## Constraints & Assumptions

- **Constraint**: Must not modify existing Minigrid source files — purely additive as a new subpackage
- **Constraint**: No continuous actions; physical action space remains discrete
- **Constraint**: Agent count is fixed per episode (no spawning/despawning)
- **Assumption**: PettingZoo is added as an optional dependency (not required for single-agent usage)
- **Assumption**: Researchers are familiar with both Gymnasium and PettingZoo APIs
- **Assumption**: The existing `Grid` class and object system are sufficient for multi-agent scenarios

## Out of Scope

- **Dialog/communication actions**: Future phase — the action space architecture will be extensible, but no language/message actions in v1
- **Heterogeneous action spaces**: All agents share the same action space
- **Continuous action spaces**: Physical actions remain discrete
- **Dynamic agent count**: No spawning or removing agents mid-episode
- **BabyAI language grounding integration**: Future phase
- **Training algorithms or baselines**: Only the environment, not the training code
- **Wrappers**: No multi-agent wrappers in v1 (e.g., reward shaping, observation stacking)
- **Agent-to-agent object passing**: Agents cannot hand objects to each other in v1

## Dependencies

- **PettingZoo** (`pettingzoo >= 1.24.0`): Required for `ParallelEnv` base class and API test
- **Existing Minigrid core**: `Grid`, `WorldObj` subclasses, `Actions` enum, constants, rendering utilities
- **Gymnasium** (`gymnasium >= 0.26.0`): Already a Minigrid dependency, used for spaces
- **NumPy, Pygame**: Already Minigrid dependencies, no new additions needed
