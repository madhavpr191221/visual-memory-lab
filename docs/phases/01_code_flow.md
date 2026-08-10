# Phase 1 code flow

## Objective

Phase 1 builds a reliable source of visual memories.

Before adding CLIP or search, the project needs a repeatable way to produce:

- images showing what an agent saw;
- the agent's location and viewing direction;
- the action that led to each image;
- the objects that were actually visible;
- complete simulator ground truth.

The output of this phase is a small dataset generator. It does not perform
retrieval yet. Phase 2 will search the images created here.

```text
controlled world
    -> reproducible movement
    -> images and metadata
    -> input for later visual-memory experiments
```

## Terms

### Environment

The simulated world, including its rooms, walls, corridor, objects, and movement
rules.

### Agent

The entity that moves through the environment. It has a grid position and a
viewing direction.

### Episode

One complete inspection run through the environment. An episode begins when the
environment is reset and ends when the scripted route finishes.

Each episode has its own seed and its own object arrangement.

### Seed

A number used to control random choices. The same seed produces the same object
placement. A different seed can produce a different placement.

This makes an experiment reproducible: seed 42 can be run again later to recreate
the same episode.

### Action

One command applied to the agent. Phase 1 uses three actions:

- `forward` moves one grid cell;
- `left` rotates the agent left;
- `right` rotates the agent right.

Turning counts as an action because it changes what the agent sees.

### Step and simulator time

A step is one point in the episode. Step 0 is the initial observation before any
action. Every action advances the step by one.

`sim_time` is logical simulator time and equals the step number. It is not a real
clock timestamp.

### Observation

One saved visual memory. It combines an RGB image with the episode, step, seed,
agent pose, previous action, and visible objects.

### Egocentric frame

An image from the agent's own point of view. It moves and rotates with the agent.
The agent cannot see the whole map, and walls can hide objects.

### Pose

The agent's position and viewing direction together. Images captured from the
same position can differ if the agent faces different directions.

### Trajectory

The ordered sequence of actions, poses, observations, and metadata produced
during an episode.

### Ground truth

Facts read directly from the simulator instead of predicted from an image. For
example, two red balls look identical, but the simulator knows which one is
`red-ball-a` and which one is `red-ball-b`.

### Manifest

A JSON file describing a run or episode, including seeds, object locations,
image dimensions, file paths, and record counts.

### JSONL

JSON Lines is a text format containing one JSON object per line. In
`observations.jsonl`, every line is one observation.

## What each episode generates

One episode uses one seed and one object arrangement. It generates:

1. **38 egocentric PNG frames**
   - frame `0000.png` is the initial view;
   - the remaining 37 frames are captured after the 37 scripted actions;
   - every frame is a 56x56 RGB image.
2. **One overview image**
   - `overview.png` is a 120x72 bird's-eye view of the complete map;
   - it is only a debugging aid and will not enter the searchable memory.
3. **One scene manifest**
   - `scene.json` records the episode seed, object identities and positions,
     route name, action count, frame count, and completion reason.
4. **38 observation records in the run-level JSONL file**
   - each record points to one egocentric frame;
   - each record stores the pose, previous action, seed, time, and visible
     objects.
5. **One episode entry in `run.json`**
   - this entry points to the episode's scene manifest and overview;
   - it also records the seed and observation count.

Therefore, an episode owns 40 files directly:

```text
38 egocentric frames + 1 overview + 1 scene manifest = 40 files
```

`run.json` and `observations.jsonl` are shared by the complete run rather than
duplicated inside every episode.

With 10 episodes, the default run produces:

```text
10 x 38 observations = 380 observation records
380 egocentric frames + 10 overview images = 390 PNG files
```

## How one episode is generated

### 1. Choose the episode seed

Episode seeds are derived from the base seed:

```text
episode-000 -> seed 42
episode-001 -> seed 43
...
episode-009 -> seed 51
```

### 2. Reset and build the scene

The reset creates a 15x9 environment containing two mirrored rooms connected by
a narrow corridor. The agent starts in the left room at position `(2, 4)`, facing
east.

Two identical red balls and one blue box are assigned to valid corner positions.
The seed controls these choices. Objects can move between episodes but remain
fixed during one episode.

The environment is implemented in
[`environment.py`](../../src/visual_memory_lab/environment.py).

### 3. Build the inspection route

The route lists positions and directions from which the agent should inspect
both rooms. A breadth-first search finds a short collision-free path between
these positions.

Breadth-first search, or BFS, explores nearby grid cells first. On this grid it
finds a shortest path around walls and objects.

The path is converted into `left`, `right`, and `forward` actions. The resulting
route contains 37 actions and deliberately observes each red ball from multiple
positions.

The route is implemented in
[`route.py`](../../src/visual_memory_lab/route.py).

### 4. Save the overview and initial observation

Before the agent moves, the generator saves the complete map as `overview.png`.
It then saves the agent's initial egocentric image as `0000.png`.

The first observation has:

```json
{
  "step": 0,
  "sim_time": 0,
  "action": null
}
```

The action is `null` because no action was needed to reach the initial state.

### 5. Execute and record every action

For each of the 37 actions, the generator:

1. moves or turns the agent;
2. receives the new egocentric RGB image;
3. checks that the episode did not end unexpectedly;
4. saves the image as the next numbered PNG;
5. records the new position and direction;
6. asks the simulator which registered objects are visible;
7. adds one record to the episode's observation list.

A `terminated` episode reached a task-ending state. A `truncated` episode hit an
external limit such as `max_steps`. Neither should happen during the 37-action
inspection route.

### 6. Write the episode manifest

After the route completes, `scene.json` records:

- `episode_id` and seed;
- map and route versions;
- the three scene objects and their positions;
- 37 actions and 38 frames;
- `route_complete` as the stop reason;
- the overview path.

### 7. Add the episode to the complete run

The episode's 38 observation records are appended to the shared
`observations.jsonl` content. A short episode entry is added to `run.json`.

Once all episodes finish, both run-level files are written. The generator first
uses a temporary directory and moves it into the requested output location only
after the run succeeds. This prevents an incomplete run from looking valid.

This orchestration is implemented in
[`trajectory.py`](../../src/visual_memory_lab/trajectory.py).

## Observation record

The observation contract is defined in
[`observations.py`](../../src/visual_memory_lab/observations.py).

A record looks like this:

```json
{
  "observation_id": "episode-000:0001",
  "episode_id": "episode-000",
  "step": 1,
  "sim_time": 1,
  "environment_seed": 42,
  "image_path": "episodes/episode-000/frames/0001.png",
  "agent_position": [3, 4],
  "agent_direction": 0,
  "agent_direction_name": "east",
  "action": "forward",
  "visible_objects": [
    {
      "object_id": "red-ball-a",
      "type": "ball",
      "color": "red",
      "state": "stationary",
      "position": [4, 6]
    }
  ]
}
```

Visibility comes from MiniGrid's simulator state. It accounts for the agent's
direction, limited field of view, and walls. It is not predicted by an image
model.

## Complete output layout

```text
phase-01-demo/
|-- run.json
|-- observations.jsonl
`-- episodes/
    |-- episode-000/
    |   |-- scene.json
    |   |-- overview.png
    |   `-- frames/
    |       |-- 0000.png
    |       `-- ...
    `-- episode-009/
```

## Code flow

```text
CLI command
    |
    v
GenerationConfig
    |
    v
generate_trajectories
    |
    +--> create episode seed
    +--> reset environment and place objects
    +--> build scripted route
    +--> save overview and step 0
    +--> execute 37 actions
    +--> save 37 more frames and records
    +--> write scene.json
    |
    v
write observations.jsonl and run.json
```

The command-line entry point is
[`cli.py`](../../src/visual_memory_lab/cli.py). It validates arguments and passes
the generation request to the trajectory module.

## What Phase 1 does not do

Phase 1 does not create embeddings, search images, calculate retrieval metrics,
or display a user interface. It establishes the input contract for those later
phases:

```text
simulator state
    + agent observation
    + time and pose
    + visible-object ground truth
    = reproducible visual-memory record
```
