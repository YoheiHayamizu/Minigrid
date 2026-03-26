---
name: multi-agent
status: backlog
created: 2026-03-25T23:12:48Z
progress: 40%
prd: .claude/prds/multi-agent.md
github: https://github.com/YoheiHayamizu/Minigrid/issues/1
---

# Epic: multi-agent

## Overview

Add a `minigrid.multigrid` subpackage that provides a PettingZoo `ParallelEnv`-compatible multi-agent base class. The implementation is purely additive — no existing Minigrid files are modified. The base class reuses the existing `Grid`, `WorldObj`, `Actions`, and rendering infrastructure, extending only where necessary (agent rendering with per-agent colors, observation generation per agent, simultaneous collision resolution).

## Architecture Decisions

### AD-1: Subpackage, not subclass of MiniGridEnv
`MultiGridEnv` extends PettingZoo's `ParallelEnv` directly rather than subclassing `MiniGridEnv`. Rationale: `MiniGridEnv` hardcodes single-agent state (`self.agent_pos`, `self.agent_dir`, `self.carrying`) throughout its methods. Wrapping it would require overriding nearly every method. Instead, we port the relevant logic (observation generation, view computation, placement helpers, random utilities) into a standalone class that composes with the same `Grid` object. This keeps the code clean and avoids fragile inheritance.

### AD-2: AgentState dataclass for per-agent state
Each agent's state is encapsulated in a frozen-field dataclass: `pos`, `dir`, `color`, `carrying`, `terminated`, `truncated`. The base class stores these in an ordered dict keyed by agent name (`"agent_0"`, etc.). This makes it trivial to iterate agents and to extend with new fields (e.g., dialog state) in future phases.

### AD-3: Simultaneous collision resolution
Movement is resolved in two passes:
1. **Intent pass**: Compute each agent's intended next position based on their forward action.
2. **Conflict pass**: If two or more agents intend to move to the same cell, or an agent intends to move to a cell currently occupied by another agent that isn't moving away, all conflicting agents stay in place.
3. **Commit pass**: Non-conflicting moves are applied.

Object interactions (pickup, drop, toggle) happen after movement is committed, processed in agent index order (deterministic). If two agents try to pick up the same object, the lower-index agent succeeds.

### AD-4: Compatibility shim for Door.toggle
`Door.toggle(env, pos)` reads `env.carrying` to check for a matching key. During action processing, the base class temporarily sets `self.carrying` to the acting agent's carried object before calling `toggle()`, then restores it. This avoids modifying `Door` or any existing `WorldObj` subclass.

### AD-5: Grid rendering with multiple agents
Rather than modifying `Grid.render()`, `MultiGridEnv` renders the grid with no agent, then overlays each agent's colored triangle tile on top. The tile rendering uses the existing `Grid.render_tile()` with a per-agent color parameter (extending the cache key). This requires a small helper but no changes to `Grid`.

### AD-6: PettingZoo as optional dependency
PettingZoo is imported at the module level inside `minigrid/multigrid/`. It is not added to the core Minigrid dependencies. Instead, it's listed as an optional extra (`pip install minigrid[multigrid]`). This ensures the single-agent package stays lightweight.

## Technical Approach

### Package Structure
```
minigrid/multigrid/
├── __init__.py              # Public API exports
├── multigrid_env.py         # MultiGridEnv base class
├── agent.py                 # AgentState dataclass
├── rendering.py             # Multi-agent rendering helpers
└── envs/
    ├── __init__.py
    ├── empty.py             # MultiGrid-Empty-v0
    ├── doorkey.py           # MultiGrid-DoorKey-v0
    └── adversarial.py       # MultiGrid-Adversarial-v0
```

### Core Class: MultiGridEnv

```python
class MultiGridEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, mission_space, num_agents, grid_size=None, width=None,
                 height=None, max_steps=100, see_through_walls=False,
                 agent_view_size=7, render_mode=None, full_obs=False,
                 agent_colors=None, ...):
        ...

    # PettingZoo API
    def reset(self, seed=None, options=None) -> tuple[dict, dict]: ...
    def step(self, actions: dict[str, int]) -> tuple[dict, dict, dict, dict, dict]: ...
    def observation_space(self, agent: str) -> spaces.Dict: ...
    def action_space(self, agent: str) -> spaces.Discrete: ...

    # Subclass hook (same pattern as MiniGridEnv)
    def _gen_grid(self, width, height): ...

    # Ported helpers
    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=inf): ...
    def put_obj(self, obj, i, j): ...
    def place_agent(self, agent_index, top=None, size=None, rand_dir=True): ...

    # Per-agent observation
    def gen_obs(self, agent_name: str) -> dict: ...
    def gen_obs_grid(self, agent_name: str) -> tuple[Grid, np.ndarray]: ...
```

### Observation Generation
- Port `MiniGridEnv.get_view_exts()`, `gen_obs_grid()`, and `gen_obs()` to take agent state as input instead of reading `self.agent_pos/dir`
- In POMDP mode: each agent gets a rotated partial view with visibility masking
- In MDP mode: each agent gets `self.grid.encode()` with all agents placed as `AgentObj` objects
- Other agents visible within an agent's view appear as colored agent objects (type index 10, color index per agent)

### Rendering
- `get_full_render()`: Render grid without agents, then stamp colored triangles for each agent
- Agent triangle color comes from `AgentState.color` instead of hardcoded red
- Highlight mask is computed per-agent (union of all agents' views, or selected agent)

### Step Logic
```
1. Increment step_count
2. Compute rotation actions (left/right) — no conflicts possible
3. Compute forward movement intents
4. Resolve collisions (conflict graph)
5. Commit movements
6. Process object interactions (pickup/drop/toggle) in agent index order
7. Check termination conditions per agent
8. Check truncation (max_steps)
9. Generate observations for active agents
10. Remove terminated/truncated agents from self.agents (PettingZoo convention)
```

### Frontend Components
N/A — this is a Python library, no frontend.

### Backend Services
N/A — no services.

### Infrastructure
- Add `pettingzoo>=1.24.0` to `pyproject.toml` as optional extra: `[multigrid]`
- Add test dependencies: `pettingzoo[testing]`
- Register example envs in `minigrid/multigrid/__init__.py`

## Implementation Strategy

Linear dependency chain for tasks 1-4, then tasks 5-8 can be parallelized.

1. **Agent state + package scaffold** — `AgentState` dataclass, package structure, `__init__.py` exports
2. **Core MultiGridEnv base class** — constructor, reset, abstract `_gen_grid`, PettingZoo properties (`possible_agents`, `agents`, spaces), random utilities (ported from MiniGridEnv)
3. **Step logic + collision resolution** — simultaneous movement, object interactions, termination/truncation, compatibility shim for `Door.toggle`
4. **Observation generation** — per-agent POMDP and MDP observation, agent visibility in grid encoding
5. **Rendering** — multi-agent full grid rendering, pygame display, rgb_array output
6. **Example env: Empty** — N agents, N goals, independent navigation
7. **Example env: DoorKey** — cooperative, one key + locked door, requires coordination
8. **Example env: Adversarial** — competitive, single goal, first agent wins
9. **Tests + PettingZoo API conformance** — unit tests, `parallel_api_test`, smoke tests
10. **Packaging + registration** — optional dependency in pyproject.toml, env registration

## Task Breakdown Preview

| # | Task | Depends | Parallel |
|---|------|---------|----------|
| 1 | Agent state dataclass + package scaffold | — | No |
| 2 | Core MultiGridEnv base class | 1 | No |
| 3 | Step logic + collision resolution | 2 | No |
| 4 | Observation generation (POMDP + MDP) | 2 | No |
| 5 | Multi-agent rendering | 2 | Yes (with 3,4) |
| 6 | Example env: MultiGrid-Empty | 3, 4 | Yes |
| 7 | Example env: MultiGrid-DoorKey | 3, 4 | Yes (with 6) |
| 8 | Example env: MultiGrid-Adversarial | 3, 4 | Yes (with 6,7) |
| 9 | Tests + PettingZoo API conformance | 6, 7, 8 | No |
| 10 | Packaging + env registration | 9 | No |

## Dependencies

- PettingZoo >= 1.24.0 (new optional dependency)
- Existing Minigrid core: `Grid`, `WorldObj`, `Actions`, constants, rendering utils
- No modifications to existing code

## Success Criteria (Technical)

1. `from minigrid.multigrid import MultiGridEnv` works
2. All 3 example envs instantiate, reset, step, and render without errors
3. PettingZoo `parallel_api_test` passes for each example env
4. `pytest tests/` (existing tests) passes with no changes
5. 2-agent Empty env steps at >= 50% throughput of single-agent Empty env

## Estimated Effort

~10 tasks, with tasks 5-8 parallelizable. Core implementation (tasks 1-4) is sequential and forms the critical path.

## Tasks Created
- [ ] #2 - Agent state dataclass + package scaffold (parallel: false)
- [ ] #3 - Core MultiGridEnv base class (parallel: false)
- [ ] #4 - Step logic + collision resolution (parallel: true)
- [ ] #5 - Observation generation POMDP + MDP (parallel: true)
- [ ] #6 - Multi-agent rendering (parallel: true)
- [ ] #7 - Example env: MultiGrid-Empty (parallel: true)
- [ ] #8 - Example env: MultiGrid-DoorKey (parallel: true)
- [ ] #9 - Example env: MultiGrid-Adversarial (parallel: true)
- [ ] #10 - Tests + PettingZoo API conformance (parallel: false)
- [ ] #11 - Packaging + environment registration (parallel: false)

Total tasks: 10
Parallel tasks: 6
Sequential tasks: 4
Estimated total effort: 27 hours
