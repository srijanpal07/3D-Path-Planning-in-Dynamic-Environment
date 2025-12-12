# 3D-Path-Planning-in-Dynamic-Environment

This repository is for the project 3D path planning in dynamic environments.
The system supports:
- Static and dynamic 3D obstacles (boxes)
- Environment generation with reusable JSON configs
- 3D occupancy grid construction
- A baseline 3D A* planner
- Interactive Plotly visualization of the planned trajectory and obstacle motion


## 1. Project Structure

```
3D-Path-Planning-in-Dynamic-Environment/
│
├── examples/
│   ├── generate_env.py          # create environment + save JSON + preview visualization
│   ├── plan_and_visualize.py    # load environment + run planner + animate visualization
│
├── world/
│   ├── schema.py                # Obstacle, Frame, Scenario dataclasses
│   ├── env_config.py            # EnvironmentConfig dataclass
│   ├── env_io.py                # JSON save/load utilities
│   ├── grid_world.py            # 3D occupancy grid
│   ├── astar3d.py               # baseline A* implementation
│
├── viz/
│   ├── plotly_viz.py            # Plotly rendering of 3D Scenario
│
├── scenes/                      # saved environments (JSON)
├── results/                     # generated visualizations (HTML)
│
├── requirements.txt
└── README.md
```


## 2. Setup Instructions

You can use either Python venv or Conda. Both instructions are provided below.

### Option A - Using Python venv (recommended)

```
python3 -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check installation:
```
python -c "import numpy, plotly; print('OK')"
```

### Option B - Using Conda
```
conda create -n path3d python=3.10
conda activate path3d
pip install -r requirements.txt
```

## 3. Running the Code
### Step 1 (optional) - Generate an Environment

This script creates the environment, writes a JSON config into the scenes/ folder, and also produces a preview visualization (viz.html).

```
python -m examples.generate_env
```

Output:
```
scenes/baseline_env.json   # environment only
viz.html                   # optional preview
```

### Step 2 (Main) - Load Environment + Plan Path + Visualize

This script loads a saved JSON environment, runs a planner from start to goal, and outputs a full animation showing obstacles + trajectory + probe motion (from start to goal).

```
python -m examples.plan_and_visualize
```

Output:
```
baseline_viz.html   # full animated visualization
```

Open the HTML file in any browser to view the scene.


## 4. Adding New Planners (optional)

Once the baseline works, you can add new algorithms: D* Lite, RRT / RRT*, Model Predictive Control (MPC), Dynamic re-planning for unknown environments. All planning algorithms only need to operate on:

```
world/grid_world.py    – occupancy map
world/schema.py        – geometry + scenario
```

## 5. Saving & Loading Environments (optional)

Environments are stored as JSON using:

```
world/env_config.py
world/env_io.py
```

A saved environement JSON file contains:

```
world bounds
start + goal locations
static obstacle boxes
dynamic obstacle trajectories (min0/max0 → min1/max1)
```

The saved JSON does NOT store the path. This allows experiments to be repeated with different planners.

## 6. Requirements
```
numpy
plotly
```