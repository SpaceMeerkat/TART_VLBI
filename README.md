# TART_VLBI

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd TART
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv ~/envs/TARTenv
source ~/envs/TARTenv/bin/activate
```

### 3. Install dependencies

```bash
pip install .
```

To also install JupyterLab and the interactive notebook extras:

```bash
pip install ".[dev]"
```

### 4. Launch JupyterLab

```bash
jupyter lab
```
