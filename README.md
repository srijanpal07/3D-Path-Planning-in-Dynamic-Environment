# 3D-Path-Planning-in-Dynamic-Environment

This repository is for the project 3D path planning in dynamic environments.
The system supports:
- Static and dynamic 3D obstacles (boxes)
- Environment generation with reusable JSON configs
- 3D occupancy grid construction
- A baseline 3D A* planner
- Interactive Plotly visualization of the planned trajectory and obstacle motion


## 1. Project Structure

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


## 2. Setup Instructions

You can use either Python venv or Conda. Both instructions are provided below.

### Option A — Using Python venv (recommended)
python3 -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate

pip install -r requirements.txt


Check installation:

python -c "import numpy, plotly; print('OK')"

## Setup
```
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the demo (produces viz.html)
python3 examples/demo_viz.py
or
python3 -m examples.demo_viz
