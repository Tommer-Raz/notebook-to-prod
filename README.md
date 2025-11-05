## Notebook to Training (CrewAI + uv)

Turn a Jupyter Notebook into a clean, reviewable training script using a small CrewAI pipeline, managed with uv.

### What this project does
- **Extracts code** from a specified `.ipynb` notebook
- **Refactors it** into a training script via a two-agent CrewAI flow
- **Outputs** two files under `output/`:
  - `output/train.py`: generated training script
  - `output/refactor.py`: reviewed/refactored version

### Architecture overview
- **Flow orchestration**: `crewai.flow.Flow` in `main.py` reads the notebook and orchestrates steps
- **Agents/Tasks**: defined in `crews/mlops_crew/crew.py` using `agents.yaml` and `tasks.yaml`
- **LLM backend**: `ollama/qwen3:1.7b` at `http://localhost:11434`

### Prerequisites
- **Python**: >= 3.11
- **uv**: for environment and dependency management
  - Install: see `https://docs.astral.sh/uv/` or:
  ```bash
  pip install uv
  ```
- **Ollama** with the model `qwen3:1.7b` locally available and running
  - Install Ollama: `https://ollama.com/download`
  - Pull model:
  ```bash
  ollama pull qwen3:1.7b
  ```
  - Ensure the server is running on `http://localhost:11434`

### Install with uv
This project is configured via `pyproject.toml`.

```bash
# From the repo root
uv sync
```

This will create/refresh the virtual environment and install:
- `crewai`
- `nbformat`

Run any script with the environment active via `uv run`:

```bash
uv run python main.py
```

### Usage
By default, `main.py` loads the notebook path hardcoded in `kickoff()`.

- To run as-is (Windows path in the repo author’s setup), simply:
```bash
uv run python main.py
```

- To process a different notebook, update the `notebook_path` argument in `main.py`:
```startLine:endLine:c:\Users\tomme\OneDrive\Desktop\jupyter to training\jupyter_to_training\main.py
29:    return flow.kickoff(inputs={"notebook_path": 
30:                                "C:\\path\\to\\your\\notebook.ipynb"})
```

After a successful run, check the generated files in `output/`.

### Configuration
- **Agents**: `crews/mlops_crew/config/agents.yaml`
- **Tasks**: `crews/mlops_crew/config/tasks.yaml`
- **Crew definition**: `crews/mlops_crew/crew.py`
- **LLM**: configured to `ollama/qwen3:1.7b` with a 1200s timeout. Adjust in `crews/mlops_crew/crew.py` if needed.

### Notes and tips
- Ensure Ollama is running before invoking the flow, otherwise the agents will fail to generate outputs.
- If you prefer a different local model, change the `LLM(model=..., base_url=...)` in `crews/mlops_crew/crew.py`.
- The example Titanic pipeline under `output/train.py`/`output/refactor.py` shows the expected style of generated code.

### Common uv commands
```bash
# Install deps / create venv
uv sync

# Run a module/script inside the uv environment
uv run python main.py

# Add a dependency (example)
uv add pandas
```

### License
Add your preferred license information here.


