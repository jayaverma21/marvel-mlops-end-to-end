<h1 align="center">
Marvelous MLOps Free End-to-end MLOps with Databricks Course

## Configuration (Required Before Running)

Before running the pipeline (or executing it manually), you **must update the catalog and schema names** to match your environment.

1. Open the configuration file:
   ```bash
   project_config_marvel.yml
   ```
2. Find the following fields:
    - catalog
    - schema

3. Replace them with your own catalog and schema values.
    ```bash
    catalog: your_catalog_name
    schema: your_schema_name
   ```
⚠️ If you do not update these values, the pipeline may fail due to incorrect or missing database targets.

### Environment Setup

This project uses **UV** for fast, reliable Python package management.

1. **Install UV** (if not already installed):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/marvelousmlops/marvel-characters.git
   cd marvel-characters
   ```

3. **Create and activate the environment**:
   ```bash
   uv sync --extra dev
   ```
   
   This command will:
   - Create a new virtual environment
   - Install all dependencies from `pyproject.toml`
   - Install development dependencies
   - Generate a lockfile (`uv.lock`)


# Data
Using the [**Marvel Characters Dataset**](https://www.kaggle.com/datasets/mohitbansal31s/marvel-characters?resource=download) from Kaggle.

This dataset contains detailed information about Marvel characters (e.g., name, powers, physical attributes, alignment, etc.).
It is used to build classification and feature engineering models for various MLOps tasks, such as predicting character attributes or status.

# Scripts

- `01.process_data.py`: Loads and preprocesses the Marvel dataset, splits into train/test, and saves to the catalog.
- `02.train_register_fe_model.py`: Performs feature engineering and trains the Marvel character model.
- `03.deploy_model.py`: Deploys the trained Marvel model to a Databricks model serving endpoint.
- `04.post_commit_status.py`: Posts status updates for Marvel integration tests to GitHub.
- `05.refresh_monitor.py`: Refreshes monitoring tables and dashboards for Marvel model serving.
