# TART_VLBI

## Easy-install

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
pip install -r requirements.txt
```

### 4. Add kernel to Jupyter Notebooks

```bash
python3 -m ipykernel install --user --name=TARTenv
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

## Slow Installation

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
