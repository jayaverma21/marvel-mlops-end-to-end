# Marvelous MLOps: End-to-End MLOps with Databricks

A complete end-to-end Machine Learning Operations (MLOps) pipeline for Marvel character data, demonstrating best practices in ML model development, deployment, and monitoring using Databricks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Scripts Overview](#scripts-overview)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project showcases a production-ready MLOps workflow using the Marvel Characters dataset. It demonstrates the complete machine learning lifecycle from data preprocessing to model deployment and monitoring on the Databricks platform. This repository serves as both a learning resource and a template for implementing MLOps best practices in real-world scenarios.

**Key Learning Areas:**
- Data preprocessing and feature engineering
- Model training and registration
- Automated deployment pipelines
- Model monitoring and performance tracking
- CI/CD integration with GitHub Actions
- Databricks serverless computing

## ✨ Features

- **Complete MLOps Pipeline**: End-to-end workflow from data ingestion to model serving
- **Databricks Integration**: Leverages Databricks serverless capabilities (version 3)
- **Automated Testing**: Integration tests with GitHub status updates
- **Model Monitoring**: Real-time monitoring dashboards and refresh mechanisms
- **Feature Engineering**: Advanced feature engineering pipeline for Marvel character attributes
- **Scalable Architecture**: Built on modern MLOps tools and best practices
- **Version Control**: Comprehensive versioning for data, models, and code
- **Production-Ready**: Deployment to Databricks model serving endpoints

## 📊 Dataset

### Marvel Characters Dataset

This project uses the **[Marvel Characters Dataset](https://www.kaggle.com/datasets/mohitbansal31s/marvel-characters?resource=download)** from Kaggle.

**Dataset Details:**
- **Source**: Kaggle
- **Content**: Detailed information about Marvel comic book characters
- **Features Include**:
  - Character names
  - Superpowers and abilities
  - Physical attributes (height, weight, etc.)
  - Alignment (hero, villain, neutral)
  - Origin stories
  - Marital status
  - Gender and identity
  - Appearance counts

**Use Cases:**
- Character attribute prediction
- Classification models (hero vs. villain)
- Feature engineering for character status prediction
- Mutant detection based on origin stories
- Character clustering and similarity analysis

## 📁 Project Structure

```
marvel-characters/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD workflows
├── data/                   # Dataset storage
├── demo_artifacts/         # Demo outputs and artifacts
├── notebooks/              # Jupyter notebooks for exploration
├── resources/              # Additional project resources
├── scripts/                # Main MLOps pipeline scripts
│   ├── 01.process_data.py
│   ├── 02.train_register_fe_model.py
│   ├── 03.deploy_model.py
│   ├── 04.post_commit_status.py
│   └── 05.refresh_monitor.py
├── src/
│   └── marvel_characters/  # Source code modules
│       ├── data_processor.py
│       └── [other modules]
├── tests/                  # Unit and integration tests
├── .gitignore
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── databricks.yml          # Databricks configuration
├── project_config_marvel.yml # Project-specific configuration
├── pyproject.toml          # Python project dependencies
├── uv.lock                 # UV lock file
├── version.txt             # Version tracking
└── README.md
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8+
- Databricks account with serverless capabilities
- UV package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- Git for version control
- Kaggle account (for dataset access)
- Databricks CLI installed
- Docker (optional, for containerized workflows)

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

4. **Download the dataset**:
   - Visit [Kaggle Marvel Characters Dataset](https://www.kaggle.com/datasets/mohitbansal31s/marvel-characters?resource=download)
   - Download and place in the `data/` directory

---

## 🔧 Databricks Configuration & Setup

### Step 1: Install Databricks CLI

```bash
# Install Databricks CLI
pip install databricks-cli

# Verify installation
databricks --version
```

### Step 2: Databricks Authentication

#### Option A: Personal Access Token (PAT) - Recommended

1. **Generate a Personal Access Token in Databricks**:
   - Log into your Databricks workspace
   - Click on your profile icon (top right) → **Settings**
   - Navigate to **Developer** → **Access Tokens**
   - Click **Generate New Token**
   - Set token name (e.g., "marvel-mlops-token")
   - Set lifetime (e.g., 90 days or as required)
   - Copy and save the token securely

2. **Configure Databricks CLI with PAT**:
   ```bash
   databricks configure --token
   ```
   
   When prompted, enter:
   - **Databricks Host**: `https://<your-workspace>.cloud.databricks.com`
   - **Token**: Paste your PAT token

#### Option B: Configuration File Method

1. **Create/Edit `.databrickscfg` file**:
   ```bash
   # Linux/macOS
   nano ~/.databrickscfg
   
   # Windows
   notepad %USERPROFILE%\.databrickscfg
   ```

2. **Add your workspace configuration**:
   ```ini
   [DEFAULT]
   host = https://<your-workspace>.cloud.databricks.com
   token = <your-personal-access-token>
   
   [dev]
   host = https://<dev-workspace>.cloud.databricks.com
   token = <dev-token>
   
   [prod]
   host = https://<prod-workspace>.cloud.databricks.com
   token = <prod-token>
   ```

3. **Test authentication**:
   ```bash
   databricks workspace ls /
   ```

### Step 3: Install Databricks Extensions

#### VS Code Extension (Recommended for Development)

1. **Install Databricks Extension for VS Code**:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X)
   - Search for "Databricks"
   - Install the official Databricks extension

2. **Configure the extension**:
   - Open Command Palette (Ctrl+Shift+P or Cmd+Shift+P)
   - Type "Databricks: Configure Databricks"
   - Select your workspace profile
   - Authenticate using PAT

#### JetBrains PyCharm Plugin

1. **Install Databricks Plugin**:
   - File → Settings → Plugins
   - Search for "Databricks"
   - Install and restart PyCharm

### Step 4: Select and Configure Databricks Cluster

#### Create a New Cluster

```bash
# Create cluster using Databricks CLI
databricks clusters create --json-file cluster-config.json
```

**Sample `cluster-config.json`**:
```json
{
  "cluster_name": "marvel-mlops-cluster",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "i3.xlarge",
  "driver_node_type_id": "i3.xlarge",
  "num_workers": 2,
  "autoscale": {
    "min_workers": 1,
    "max_workers": 4
  },
  "spark_conf": {
    "spark.databricks.delta.preview.enabled": "true"
  },
  "custom_tags": {
    "Project": "MarvelMLOps",
    "Environment": "Development"
  }
}
```

#### List Available Clusters

```bash
# List all clusters
databricks clusters list

# Get cluster details
databricks clusters get --cluster-id <cluster-id>
```

#### Start/Stop Cluster

```bash
# Start cluster
databricks clusters start --cluster-id <cluster-id>

# Stop cluster
databricks clusters stop --cluster-id <cluster-id>

# Restart cluster
databricks clusters restart --cluster-id <cluster-id>
```

### Step 5: Setup Python Environment in Databricks

#### Option A: Using Databricks Connect (Local Development)

1. **Install Databricks Connect**:
   ```bash
   pip install databricks-connect==13.3.*
   ```

2. **Configure Databricks Connect**:
   ```bash
   databricks-connect configure
   ```
   
   Enter the following information:
   - **Databricks Host**: `https://<workspace>.cloud.databricks.com`
   - **Token**: Your PAT token
   - **Cluster ID**: Your cluster ID
   - **Org ID**: Your organization ID (for AWS/Azure)
   - **Port**: 15001 (default)

3. **Test the connection**:
   ```python
   from databricks.connect import DatabricksSession
   
   spark = DatabricksSession.builder.getOrCreate()
   spark.range(10).show()
   ```

#### Option B: Using Databricks Notebooks

1. **Upload notebooks to workspace**:
   ```bash
   databricks workspace import_dir \
     ./notebooks \
     /Workspace/Users/<your-email>/marvel-mlops/notebooks \
     --overwrite
   ```

2. **Install project dependencies on cluster**:
   ```bash
   # Create init script
   cat > install_dependencies.sh << EOF
   #!/bin/bash
   pip install -r /dbfs/requirements.txt
   EOF
   
   # Upload to DBFS
   databricks fs cp install_dependencies.sh \
     dbfs:/databricks/init_scripts/install_dependencies.sh
   ```

### Step 6: Configure Databricks Workspace

#### Setup Project Structure

```bash
# Create workspace directories
databricks workspace mkdirs /Workspace/Users/<your-email>/marvel-mlops
databricks workspace mkdirs /Workspace/Users/<your-email>/marvel-mlops/notebooks
databricks workspace mkdirs /Workspace/Users/<your-email>/marvel-mlops/scripts

# Upload project files
databricks workspace import_dir \
  ./scripts \
  /Workspace/Users/<your-email>/marvel-mlops/scripts \
  --overwrite
```

#### Setup DBFS (Databricks File System)

```bash
# Create DBFS directories
databricks fs mkdirs dbfs:/FileStore/marvel-mlops/data
databricks fs mkdirs dbfs:/FileStore/marvel-mlops/models
databricks fs mkdirs dbfs:/FileStore/marvel-mlops/artifacts

# Upload data files
databricks fs cp ./data/marvel_characters.csv \
  dbfs:/FileStore/marvel-mlops/data/marvel_characters.csv \
  --overwrite

# List DBFS contents
databricks fs ls dbfs:/FileStore/marvel-mlops/
```

### Step 7: Setup Databricks Jobs and Pipelines

#### Create a Databricks Job

```bash
# Create job using JSON configuration
databricks jobs create --json-file job-config.json
```

**Sample `job-config.json`**:
```json
{
  "name": "Marvel MLOps Pipeline",
  "email_notifications": {
    "on_failure": ["your-email@example.com"]
  },
  "timeout_seconds": 3600,
  "max_concurrent_runs": 1,
  "tasks": [
    {
      "task_key": "process_data",
      "description": "Process and prepare Marvel dataset",
      "existing_cluster_id": "<cluster-id>",
      "python_wheel_task": {
        "package_name": "marvel_characters",
        "entry_point": "process_data"
      },
      "libraries": [
        {
          "pypi": {
            "package": "pandas==2.0.3"
          }
        }
      ]
    },
    {
      "task_key": "train_model",
      "description": "Train and register model",
      "depends_on": [
        {
          "task_key": "process_data"
        }
      ],
      "existing_cluster_id": "<cluster-id>",
      "python_wheel_task": {
        "package_name": "marvel_characters",
        "entry_point": "train_model"
      }
    },
    {
      "task_key": "deploy_model",
      "description": "Deploy model to serving endpoint",
      "depends_on": [
        {
          "task_key": "train_model"
        }
      ],
      "existing_cluster_id": "<cluster-id>",
      "python_wheel_task": {
        "package_name": "marvel_characters",
        "entry_point": "deploy_model"
      }
    }
  ]
}
```

#### List and Manage Jobs

```bash
# List all jobs
databricks jobs list

# Get job details
databricks jobs get --job-id <job-id>

# Run job now
databricks jobs run-now --job-id <job-id>

# Get run status
databricks runs get --run-id <run-id>

# Cancel a running job
databricks runs cancel --run-id <run-id>
```

---

## 🏗️ Building and Deploying with Databricks Bundles

### What are Databricks Bundles?

Databricks Bundles (formerly Databricks Asset Bundles) allow you to define, deploy, and manage Databricks resources as code using YAML configuration files.

### Step 1: Install Databricks CLI (Bundle Support)

```bash
# Ensure you have the latest Databricks CLI
pip install --upgrade databricks-cli

# Verify bundle support
databricks bundle --help
```

### Step 2: Initialize Bundle Configuration

The `databricks.yml` file is your bundle configuration:

```yaml
bundle:
  name: marvel-mlops

workspace:
  host: https://<your-workspace>.cloud.databricks.com
  root_path: /Workspace/Users/${workspace.current_user.userName}/.bundle/${bundle.name}/${bundle.target}

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: https://<dev-workspace>.cloud.databricks.com
  
  staging:
    mode: development
    workspace:
      host: https://<staging-workspace>.cloud.databricks.com
  
  prod:
    mode: production
    workspace:
      host: https://<prod-workspace>.cloud.databricks.com
    run_as:
      service_principal_name: marvel-mlops-sp

resources:
  jobs:
    marvel_mlops_pipeline:
      name: Marvel MLOps Pipeline - ${bundle.target}
      
      tasks:
        - task_key: process_data
          notebook_task:
            notebook_path: ./notebooks/01_process_data.py
          existing_cluster_id: ${var.cluster_id}
        
        - task_key: train_model
          notebook_task:
            notebook_path: ./notebooks/02_train_model.py
          depends_on:
            - task_key: process_data
          existing_cluster_id: ${var.cluster_id}
        
        - task_key: deploy_model
          notebook_task:
            notebook_path: ./notebooks/03_deploy_model.py
          depends_on:
            - task_key: train_model
          existing_cluster_id: ${var.cluster_id}
      
      schedule:
        quartz_cron_expression: "0 0 2 * * ?"
        timezone_id: "America/New_York"

  pipelines:
    marvel_dlt_pipeline:
      name: Marvel DLT Pipeline - ${bundle.target}
      storage: /mnt/marvel-mlops/${bundle.target}/dlt
      target: marvel_${bundle.target}
      libraries:
        - notebook:
            path: ./notebooks/dlt_pipeline.py

variables:
  cluster_id:
    description: Databricks cluster ID
    default: <your-cluster-id>
```

### Step 3: Validate Bundle Configuration

```bash
# Validate the bundle configuration
databricks bundle validate

# Validate for specific target
databricks bundle validate --target prod
```

### Step 4: Deploy Bundle

```bash
# Deploy to development environment
databricks bundle deploy --target dev

# Deploy to staging
databricks bundle deploy --target staging

# Deploy to production
databricks bundle deploy --target prod
```

### Step 5: Run Bundle Jobs

```bash
# Run the entire pipeline
databricks bundle run marvel_mlops_pipeline --target dev

# Run specific task
databricks bundle run marvel_mlops_pipeline --target dev --task process_data
```

### Step 6: Destroy Bundle (Cleanup)

```bash
# Remove all deployed resources
databricks bundle destroy --target dev

# Force destroy without confirmation
databricks bundle destroy --target dev --auto-approve
```

---

## 🔄 GitHub Actions CI/CD Pipeline Setup

### Setup GitHub Secrets

Add the following secrets to your GitHub repository:
- Go to **Settings** → **Secrets and variables** → **Actions**
- Add new repository secrets:

```
DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
DATABRICKS_TOKEN=<your-pat-token>
DATABRICKS_CLUSTER_ID=<your-cluster-id>
```

### Sample GitHub Actions Workflow

Create `.github/workflows/databricks-deploy.yml`:

```yaml
name: Databricks MLOps Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install databricks-cli
          pip install -r requirements.txt
      
      - name: Validate bundle
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: |
          databricks bundle validate

  deploy:
    needs: validate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to Databricks
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: |
          pip install databricks-cli
          databricks bundle deploy --target prod
      
      - name: Run pipeline
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: |
          databricks bundle run marvel_mlops_pipeline --target prod
```

---

## 🐍 Python Environment Activation & Databricks Connect

### Local Python Environment Setup

```bash
# Activate UV environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Verify Python environment
which python
python --version

# Install Databricks Connect
pip install databricks-connect==13.3.*
```

### Configure Databricks Connect

```bash
# Interactive configuration
databricks-connect configure

# Or set environment variables
export DATABRICKS_HOST="https://<workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="<your-pat-token>"
export DATABRICKS_CLUSTER_ID="<cluster-id>"
```

### Test Databricks Connect

```python
# test_connection.py
from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession

# Create Databricks session
spark = DatabricksSession.builder.getOrCreate()

# Test with simple query
df = spark.range(10)
df.show()

# Test with Marvel data
df = spark.read.csv(
    "dbfs:/FileStore/marvel-mlops/data/marvel_characters.csv",
    header=True,
    inferSchema=True
)
df.show(5)
print(f"Total characters: {df.count()}")
```

Run the test:
```bash
python test_connection.py
```

---

## 📋 Complete Deployment Checklist

### Pre-Deployment

- [ ] Install Databricks CLI
- [ ] Configure authentication (PAT token)
- [ ] Create/select Databricks cluster
- [ ] Install Databricks Connect locally
- [ ] Test connection to Databricks workspace
- [ ] Upload data to DBFS
- [ ] Configure `databricks.yml` bundle file
- [ ] Setup GitHub secrets (for CI/CD)

### Deployment Steps

- [ ] Validate bundle configuration
- [ ] Deploy bundle to dev environment
- [ ] Test jobs in dev environment
- [ ] Deploy to staging (if applicable)
- [ ] Run integration tests
- [ ] Deploy to production
- [ ] Monitor job execution
- [ ] Setup monitoring dashboards

### Post-Deployment

- [ ] Verify model registration in MLflow
- [ ] Test model serving endpoint
- [ ] Setup alerts and monitoring
- [ ] Document deployment process
- [ ] Train team on new workflows

## 📝 Scripts Overview

### 1. **Data Processing** (`01.process_data.py`)

**Purpose**: Initial data preprocessing and preparation

**Key Functions**:
- Loads the Marvel characters dataset
- Performs data cleaning and validation
- Handles missing values and outliers
- Creates train/test splits (80/20)
- Saves processed data to Databricks catalog

**Data Transformations**:
- Gender normalization (Male, Female, Other)
- Marital status filtering
- Mutant feature extraction from origin stories
- Physical attribute validation

**Local Development Example**:
```python
# scripts/01.process_data.py
import pandas as pd
from pathlib import Path

def load_marvel_data(file_path: str) -> pd.DataFrame:
    """Load Marvel characters dataset"""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} characters")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data"""
    # Remove duplicates
    df = df.drop_duplicates(subset=['name'])
    
    # Handle missing values
    df['height'] = df['height'].fillna(df['height'].median())
    df['weight'] = df['weight'].fillna(df['weight'].median())
    
    # Normalize gender
    df['gender'] = df['gender'].map({
        'Male': 'Male',
        'Female': 'Female',
        'Other': 'Other'
    }).fillna('Unknown')
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features"""
    # Extract mutant status from origin
    df['is_mutant'] = df['origin'].str.contains('mutant', case=False, na=False)
    
    # BMI calculation
    df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
    
    return df

if __name__ == "__main__":
    # Local execution
    data_path = Path("data/marvel_characters.csv")
    df = load_marvel_data(data_path)
    df = clean_data(df)
    df = create_features(df)
    
    # Save processed data
    output_path = Path("data/processed/marvel_characters_clean.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} processed characters to {output_path}")
```

**Running Locally**:
```bash
# Run data processing script
python scripts/01.process_data.py

# With custom parameters
python scripts/01.process_data.py --input data/raw.csv --output data/processed.csv
```

**Running on Databricks**:
```python
# In Databricks notebook
%run ./scripts/01.process_data

# Or using Databricks CLI
databricks workspace import ./scripts/01.process_data.py \
  /Workspace/Users/<email>/marvel-mlops/scripts/01.process_data.py

databricks jobs run-now --job-id <job-id>
```

### 2. **Model Training & Registration** (`02.train_register_fe_model.py`)

**Purpose**: Feature engineering and model training pipeline

**Key Functions**:
- Advanced feature engineering on Marvel character data
- Model training with hyperparameter optimization
- Model evaluation and validation
- Model registration to MLflow registry
- Feature importance analysis

**Features Created**:
- Encoded categorical variables
- Scaled numerical features
- Derived mutant indicator
- Interaction features

**Local Development Example**:
```python
# scripts/02.train_register_fe_model.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering"""
    # Encode categorical variables
    le = LabelEncoder()
    df['gender_encoded'] = le.fit_transform(df['gender'])
    df['alignment_encoded'] = le.fit_transform(df['alignment'])
    
    # Create interaction features
    df['power_density'] = df['total_powers'] / (df['height'] * df['weight'])
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ['height', 'weight', 'bmi', 'total_powers']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def train_model(X_train, y_train, X_test, y_test):
    """Train Random Forest model with MLflow tracking"""
    
    with mlflow.start_run(run_name="marvel-rf-model"):
        # Log parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        return model

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("data/processed/marvel_characters_clean.csv")
    
    # Feature engineering
    df = engineer_features(df)
    
    # Prepare features and target
    feature_cols = ['height', 'weight', 'bmi', 'gender_encoded', 
                    'alignment_encoded', 'total_powers', 'is_mutant']
    X = df[feature_cols]
    y = df['target']  # e.g., 'is_hero'
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train and register model
    model = train_model(X_train, y_train, X_test, y_test)
    
    print("✅ Model training complete!")
```

**Running Locally with MLflow**:
```bash
# Start MLflow UI
mlflow ui --port 5000

# Run training script
python scripts/02.train_register_fe_model.py

# View results at http://localhost:5000
```

**Registering Model to MLflow Registry**:
```python
import mlflow

# Register the best model
model_uri = "runs:/<run-id>/model"
mlflow.register_model(model_uri, "marvel-character-classifier")

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="marvel-character-classifier",
    version=1,
    stage="Production"
)
```

### 3. **Model Deployment** (`03.deploy_model.py`)

**Purpose**: Automated model deployment to production

**Key Functions**:
- Deploys trained model to Databricks model serving endpoint
- Configures serving infrastructure
- Sets up model versioning
- Validates deployment health
- Enables real-time inference API

**Deployment Modes**:
- Serverless endpoints
- Auto-scaling configuration
- Traffic splitting for A/B testing

**Local Development Example**:
```python
# scripts/03.deploy_model.py
import mlflow
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

def deploy_to_endpoint(model_name: str, model_version: int):
    """Deploy model to Databricks serving endpoint"""
    
    w = WorkspaceClient()
    
    endpoint_config = EndpointCoreConfigInput(
        name=f"{model_name}-endpoint",
        served_entities=[
            ServedEntityInput(
                entity_name=model_name,
                entity_version=str(model_version),
                scale_to_zero_enabled=True,
                workload_size="Small"
            )
        ]
    )
    
    # Create or update endpoint
    try:
        w.serving_endpoints.create(
            name=endpoint_config.name,
            config=endpoint_config
        )
        print(f"✅ Endpoint {endpoint_config.name} created successfully!")
    except Exception as e:
        print(f"Endpoint exists, updating... {e}")
        w.serving_endpoints.update_config(
            name=endpoint_config.name,
            served_entities=endpoint_config.served_entities
        )

def test_endpoint(endpoint_name: str, sample_data: dict):
    """Test the deployed endpoint"""
    
    url = f"https://<workspace>.cloud.databricks.com/serving-endpoints/{endpoint_name}/invocations"
    
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=sample_data, headers=headers)
    
    if response.status_code == 200:
        print(f"✅ Prediction: {response.json()}")
    else:
        print(f"❌ Error: {response.text}")

if __name__ == "__main__":
    # Deploy latest model version
    deploy_to_endpoint("marvel-character-classifier", model_version=1)
    
    # Test with sample data
    sample = {
        "dataframe_records": [{
            "height": 180,
            "weight": 80,
            "bmi": 24.7,
            "gender_encoded": 0,
            "alignment_encoded": 1,
            "total_powers": 15,
            "is_mutant": True
        }]
    }
    
    test_endpoint("marvel-character-classifier-endpoint", sample)
```

**Local Testing with Flask (Mock Endpoint)**:
```python
# local_serve.py - For local testing before Databricks deployment
from flask import Flask, request, jsonify
import mlflow
import pandas as pd

app = Flask(__name__)

# Load model
model = mlflow.sklearn.load_model("models:/marvel-character-classifier/Production")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data['dataframe_records'])
    predictions = model.predict(df)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

```bash
# Run local server
python local_serve.py

# Test locally
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"height": 180, "weight": 80, ...}]}'
```

### 4. **GitHub Integration** (`04.post_commit_status.py`)

**Purpose**: CI/CD pipeline integration

**Key Functions**:
- Posts integration test results to GitHub
- Updates commit status checks
- Provides deployment feedback
- Enables automated workflows

**Status Updates**:
- Success/failure notifications
- Test coverage reports
- Deployment confirmations

**Local Development Example**:
```python
# scripts/04.post_commit_status.py
import os
import requests

def post_github_status(commit_sha: str, state: str, description: str):
    """Post status to GitHub commit"""
    
    github_token = os.getenv('GITHUB_TOKEN')
    repo = "marvelousmlops/marvel-characters"
    
    url = f"https://api.github.com/repos/{repo}/statuses/{commit_sha}"
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    payload = {
        "state": state,  # pending, success, error, failure
        "description": description,
        "context": "databricks/integration-tests"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 201:
        print(f"✅ Status posted: {state}")
    else:
        print(f"❌ Failed to post status: {response.text}")

if __name__ == "__main__":
    import sys
    
    commit_sha = sys.argv[1] if len(sys.argv) > 1 else os.getenv('GIT_COMMIT')
    
    # Run tests and post results
    test_passed = True  # Replace with actual test results
    
    if test_passed:
        post_github_status(commit_sha, "success", "All tests passed!")
    else:
        post_github_status(commit_sha, "failure", "Tests failed")
```

**Integration with CI/CD**:
```bash
# In GitHub Actions workflow
python scripts/04.post_commit_status.py ${{ github.sha }}
```

### 5. **Monitoring & Refresh** (`05.refresh_monitor.py`)

**Purpose**: Model performance monitoring and dashboard updates

**Key Functions**:
- Refreshes monitoring tables
- Updates performance dashboards
- Tracks model drift
- Monitors data quality
- Generates alerts for anomalies

**Monitoring Metrics**:
- Prediction accuracy
- Latency measurements
- Input data distribution
- Feature drift detection

**Local Development Example**:
```python
# scripts/05.refresh_monitor.py
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_drift(reference_data: pd.DataFrame, 
                   current_data: pd.DataFrame,
                   features: list) -> dict:
    """Calculate feature drift metrics"""
    
    drift_metrics = {}
    
    for feature in features:
        ref_mean = reference_data[feature].mean()
        curr_mean = current_data[feature].mean()
        
        ref_std = reference_data[feature].std()
        curr_std = current_data[feature].std()
        
        # Calculate drift score (normalized difference)
        drift_score = abs(curr_mean - ref_mean) / ref_std if ref_std > 0 else 0
        
        drift_metrics[feature] = {
            'drift_score': drift_score,
            'ref_mean': ref_mean,
            'curr_mean': curr_mean,
            'alert': drift_score > 2.0  # Alert if drift > 2 std devs
        }
    
    return drift_metrics

def monitor_model_performance(predictions_df: pd.DataFrame):
    """Monitor model predictions and performance"""
    
    metrics = {
        'total_predictions': len(predictions_df),
        'avg_confidence': predictions_df['confidence'].mean(),
        'prediction_distribution': predictions_df['prediction'].value_counts().to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Check for anomalies
    low_confidence = predictions_df[predictions_df['confidence'] < 0.5]
    if len(low_confidence) > 0:
        metrics['low_confidence_count'] = len(low_confidence)
        metrics['alert'] = True
    
    return metrics

def create_monitoring_dashboard(metrics_df: pd.DataFrame):
    """Create monitoring visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy over time
    axes[0, 0].plot(metrics_df['timestamp'], metrics_df['accuracy'])
    axes[0, 0].set_title('Model Accuracy Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Accuracy')
    
    # Prediction distribution
    axes[0, 1].bar(range(len(metrics_df['pred_dist'])), 
                   metrics_df['pred_dist'].values())
    axes[0, 1].set_title('Prediction Distribution')
    
    # Latency
    axes[1, 0].plot(metrics_df['timestamp'], metrics_df['latency_ms'])
    axes[1, 0].set_title('Inference Latency')
    axes[1, 0].set_ylabel('Latency (ms)')
    
    # Feature drift
    sns.heatmap(metrics_df[['feature1_drift', 'feature2_drift']], 
                ax=axes[1, 1], cmap='RdYlGn_r')
    axes[1, 1].set_title('Feature Drift Heatmap')
    
    plt.tight_layout()
    plt.savefig('monitoring_dashboard.png')
    print("✅ Dashboard saved to monitoring_dashboard.png")

if __name__ == "__main__":
    # Load reference and current data
    reference_data = pd.read_csv("data/reference/baseline.csv")
    current_data = pd.read_csv("data/current/latest_predictions.csv")
    
    # Calculate drift
    features = ['height', 'weight', 'bmi', 'total_powers']
    drift_metrics = calculate_drift(reference_data, current_data, features)
    
    # Check for alerts
    alerts = [f for f, m in drift_metrics.items() if m['alert']]
    if alerts:
        print(f"⚠️  ALERT: Drift detected in features: {alerts}")
    else:
        print("✅ No significant drift detected")
    
    # Monitor performance
    predictions = pd.read_csv("data/predictions/recent.csv")
    performance = monitor_model_performance(predictions)
    
    print(f"📊 Monitoring Summary:")
    print(f"  Total Predictions: {performance['total_predictions']}")
    print(f"  Avg Confidence: {performance['avg_confidence']:.3f}")
```

**Running Monitoring Locally**:
```bash
# Run monitoring script
python scripts/05.refresh_monitor.py

# Schedule with cron (Linux/Mac)
# Run every hour
0 * * * * cd /path/to/project && python scripts/05.refresh_monitor.py

# Or use Python scheduler
python -c "
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess

scheduler = BlockingScheduler()

@scheduler.scheduled_job('interval', hours=1)
def run_monitoring():
    subprocess.run(['python', 'scripts/05.refresh_monitor.py'])

scheduler.start()
"
```

---

## 🏠 Local Development Best Practices

### Setting Up Local Environment

```bash
# 1. Clone repository
git clone https://github.com/marvelousmlops/marvel-characters.git
cd marvel-characters

# 2. Create virtual environment with UV
uv sync --extra dev

# 3. Activate environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 4. Install pre-commit hooks
pre-commit install

# 5. Setup local MLflow
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow ui --port 5000 &

# 6. Create local data directories
mkdir -p data/{raw,processed,reference,current,predictions}
mkdir -p models
mkdir -p logs
```

### Local Testing Workflow

```bash
# 1. Process data locally
python scripts/01.process_data.py

# 2. Train model locally with MLflow tracking
python scripts/02.train_register_fe_model.py

# 3. Test model serving locally
python local_serve.py &

# 4. Run integration tests
pytest tests/ -v

# 5. Check code quality
pre-commit run --all-files

# 6. Run monitoring
python scripts/05.refresh_monitor.py
```

### Local to Databricks Migration

```python
# local_to_databricks.py
"""
Script to migrate from local development to Databricks
"""
import os
from databricks.sdk import WorkspaceClient

def migrate_to_databricks():
    w = WorkspaceClient()
    
    # 1. Upload data to DBFS
    os.system("""
        databricks fs cp -r data/ dbfs:/FileStore/marvel-mlops/data/
    """)
    
    # 2. Upload notebooks
    os.system("""
        databricks workspace import_dir notebooks/ \
          /Workspace/Users/${USER}/marvel-mlops/notebooks/
    """)
    
    # 3. Create job
    os.system("""
        databricks jobs create --json-file job-config.json
    """)
    
    # 4. Migrate MLflow experiments
    os.system("""
        mlflow experiments create --experiment-name /Users/${USER}/marvel-mlops
    """)
    
    print("✅ Migration to Databricks complete!")

if __name__ == "__main__":
    migrate_to_databricks()
```

### Jupyter Notebook Development

```python
# notebooks/exploration.ipynb

# Cell 1: Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Load data
df = pd.read_csv("../data/marvel_characters.csv")

# Cell 2: EDA
df.head()
df.describe()
df.info()

# Cell 3: Visualization
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='alignment')
plt.title('Character Alignment Distribution')
plt.show()

# Cell 4: Feature Engineering
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

# Cell 5: Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df[['height', 'weight', 'gender_encoded']]
y = df['is_hero']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Databricks**: Unified analytics platform
- **MLflow**: Experiment tracking and model registry
- **UV**: Fast Python package manager

### ML/Data Science
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computing
- **PySpark**: Distributed data processing

### MLOps Tools
- **Databricks Feature Store**: Feature management
- **Databricks Model Serving**: Real-time inference
- **GitHub Actions**: CI/CD automation
- **pre-commit**: Code quality hooks

### Monitoring & Visualization
- **Databricks Dashboards**: Monitoring visualization
- **MLflow UI**: Experiment tracking interface

## 📈 Usage Examples

### Running the Complete Pipeline

```bash
# 1. Process and prepare data
python scripts/01.process_data.py

# 2. Train and register model
python scripts/02.train_register_fe_model.py

# 3. Deploy to production
python scripts/03.deploy_model.py

# 4. Update GitHub status
python scripts/04.post_commit_status.py

# 5. Refresh monitoring
python scripts/05.refresh_monitor.py
```

### Configuration

Edit `project_config_marvel.yml` to customize:
- Data paths
- Model parameters
- Deployment settings
- Monitoring thresholds

## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/marvel_characters
```

---

## 📚 Databricks Commands Reference

### Workspace Commands

```bash
# List workspace contents
databricks workspace ls /Workspace/Users/<your-email>

# Export notebook
databricks workspace export \
  /Workspace/Users/<your-email>/notebook.py \
  ./local_notebook.py

# Import notebook
databricks workspace import \
  ./local_notebook.py \
  /Workspace/Users/<your-email>/notebook.py \
  --language PYTHON \
  --format SOURCE

# Import directory recursively
databricks workspace import_dir \
  ./notebooks \
  /Workspace/Users/<your-email>/marvel-mlops/notebooks \
  --overwrite

# Delete notebook/directory
databricks workspace delete /Workspace/Users/<your-email>/old_notebook.py
databricks workspace delete /Workspace/Users/<your-email>/old_folder --recursive
```

### DBFS (Databricks File System) Commands

```bash
# List DBFS contents
databricks fs ls dbfs:/
databricks fs ls dbfs:/FileStore/marvel-mlops/

# Copy file to DBFS
databricks fs cp ./local_file.csv dbfs:/FileStore/marvel-mlops/data/file.csv
databricks fs cp --recursive ./local_dir dbfs:/FileStore/marvel-mlops/data/

# Copy file from DBFS to local
databricks fs cp dbfs:/FileStore/marvel-mlops/data/file.csv ./local_file.csv

# Create directory
databricks fs mkdirs dbfs:/FileStore/marvel-mlops/new_folder

# Remove file/directory
databricks fs rm dbfs:/FileStore/marvel-mlops/old_file.csv
databricks fs rm --recursive dbfs:/FileStore/marvel-mlops/old_folder/

# Move file
databricks fs mv \
  dbfs:/FileStore/marvel-mlops/old_location/file.csv \
  dbfs:/FileStore/marvel-mlops/new_location/file.csv
```

### Cluster Commands

```bash
# List all clusters
databricks clusters list

# Get cluster information
databricks clusters get --cluster-id <cluster-id>

# Create cluster from JSON config
databricks clusters create --json-file cluster-config.json

# Create cluster from JSON string
databricks clusters create --json '{
  "cluster_name": "test-cluster",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "i3.xlarge",
  "num_workers": 2
}'

# Edit/Update cluster
databricks clusters edit --json-file updated-cluster-config.json

# Start cluster
databricks clusters start --cluster-id <cluster-id>

# Stop cluster
databricks clusters stop --cluster-id <cluster-id>

# Restart cluster
databricks clusters restart --cluster-id <cluster-id>

# Delete/Terminate cluster
databricks clusters delete --cluster-id <cluster-id>

# Get cluster events
databricks clusters events --cluster-id <cluster-id>

# List cluster node types
databricks clusters list-node-types

# List Spark versions
databricks clusters spark-versions
```

### Jobs Commands

```bash
# List all jobs
databricks jobs list

# Get job details
databricks jobs get --job-id <job-id>

# Create job from JSON config
databricks jobs create --json-file job-config.json

# Update existing job
databricks jobs reset --job-id <job-id> --json-file updated-job-config.json

# Delete job
databricks jobs delete --job-id <job-id>

# Run job immediately
databricks jobs run-now --job-id <job-id>

# Run job with parameters
databricks jobs run-now --job-id <job-id> --notebook-params '{"param1": "value1"}'

# List job runs
databricks runs list --job-id <job-id>

# Get run details
databricks runs get --run-id <run-id>

# Get run output
databricks runs get-output --run-id <run-id>

# Cancel run
databricks runs cancel --run-id <run-id>

# Submit one-time run
databricks runs submit --json-file one-time-run.json
```

### Secrets Commands

```bash
# Create secret scope
databricks secrets create-scope --scope marvel-mlops

# List secret scopes
databricks secrets list-scopes

# Put secret value
databricks secrets put --scope marvel-mlops --key api-token

# Put secret from file
databricks secrets put --scope marvel-mlops --key private-key --binary-file ./key.pem

# List secrets in scope
databricks secrets list --scope marvel-mlops

# Delete secret
databricks secrets delete --scope marvel-mlops --key old-token

# Delete scope
databricks secrets delete-scope --scope old-scope

# Get secret ACLs
databricks secrets list-acls --scope marvel-mlops

# Put secret ACL
databricks secrets put-acl --scope marvel-mlops --principal user@example.com --permission READ
```

### Libraries Commands

```bash
# Install library on cluster
databricks libraries install \
  --cluster-id <cluster-id> \
  --pypi-package pandas==2.0.3

# Install from Maven
databricks libraries install \
  --cluster-id <cluster-id> \
  --maven-coordinates org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0

# Install from Jar
databricks libraries install \
  --cluster-id <cluster-id> \
  --jar dbfs:/FileStore/jars/custom-library.jar

# List cluster libraries
databricks libraries list --cluster-id <cluster-id>

# Uninstall library
databricks libraries uninstall \
  --cluster-id <cluster-id> \
  --pypi-package pandas
```

### MLflow Commands (via Databricks)

```bash
# List experiments
databricks experiments list

# Create experiment
databricks experiments create --experiment-name "/Users/<email>/marvel-mlops-experiment"

# Get experiment
databricks experiments get --experiment-id <experiment-id>

# Delete experiment
databricks experiments delete --experiment-id <experiment-id>

# List runs in experiment
databricks runs list --experiment-id <experiment-id>
```

### Repos (Git Integration) Commands

```bash
# Create repo
databricks repos create \
  --url https://github.com/marvelousmlops/marvel-characters \
  --provider gitHub \
  --path /Repos/<your-email>/marvel-characters

# List repos
databricks repos list

# Get repo info
databricks repos get --repo-id <repo-id>

# Update repo (pull latest)
databricks repos update --repo-id <repo-id> --branch main

# Delete repo
databricks repos delete --repo-id <repo-id>
```

### Bundle Commands (Databricks Asset Bundles)

```bash
# Initialize new bundle
databricks bundle init

# Validate bundle configuration
databricks bundle validate
databricks bundle validate --target dev

# Deploy bundle
databricks bundle deploy
databricks bundle deploy --target dev
databricks bundle deploy --target prod

# Run bundle resource
databricks bundle run <job-name>
databricks bundle run <job-name> --target dev

# Destroy bundle resources
databricks bundle destroy
databricks bundle destroy --target dev --auto-approve

# Generate bundle schema
databricks bundle schema
```

### Token Management Commands

```bash
# Create token
databricks tokens create --comment "CI/CD token" --lifetime-seconds 7776000

# List tokens
databricks tokens list

# Revoke token
databricks tokens delete --token-id <token-id>
```

### Instance Pool Commands

```bash
# Create instance pool
databricks instance-pools create --json-file pool-config.json

# List instance pools
databricks instance-pools list

# Get pool details
databricks instance-pools get --instance-pool-id <pool-id>

# Edit instance pool
databricks instance-pools edit --json-file updated-pool-config.json

# Delete instance pool
databricks instance-pools delete --instance-pool-id <pool-id>
```

### SQL Warehouse Commands

```bash
# List SQL warehouses
databricks sql warehouses list

# Get warehouse details
databricks sql warehouses get --id <warehouse-id>

# Start warehouse
databricks sql warehouses start --id <warehouse-id>

# Stop warehouse
databricks sql warehouses stop --id <warehouse-id>
```

### Common Workflow Examples

#### Example 1: Complete Project Upload

```bash
#!/bin/bash
# upload_project.sh

WORKSPACE_USER="your-email@example.com"
PROJECT_PATH="/Workspace/Users/${WORKSPACE_USER}/marvel-mlops"

# Create workspace structure
databricks workspace mkdirs ${PROJECT_PATH}/notebooks
databricks workspace mkdirs ${PROJECT_PATH}/scripts

# Upload notebooks
databricks workspace import_dir \
  ./notebooks \
  ${PROJECT_PATH}/notebooks \
  --overwrite

# Upload scripts
databricks workspace import_dir \
  ./scripts \
  ${PROJECT_PATH}/scripts \
  --overwrite

# Create DBFS directories
databricks fs mkdirs dbfs:/FileStore/marvel-mlops/data
databricks fs mkdirs dbfs:/FileStore/marvel-mlops/models
databricks fs mkdirs dbfs:/FileStore/marvel-mlops/artifacts

# Upload data
databricks fs cp --recursive \
  ./data \
  dbfs:/FileStore/marvel-mlops/data \
  --overwrite

echo "✅ Project uploaded successfully!"
```

#### Example 2: Automated Deployment Script

```bash
#!/bin/bash
# deploy.sh

set -e

TARGET=${1:-dev}

echo "🚀 Deploying to ${TARGET} environment..."

# Validate bundle
echo "📋 Validating bundle..."
databricks bundle validate --target ${TARGET}

# Deploy bundle
echo "🔧 Deploying bundle..."
databricks bundle deploy --target ${TARGET}

# Run tests
echo "🧪 Running tests..."
databricks bundle run integration_tests --target ${TARGET}

# Deploy to production if all tests pass
if [ "${TARGET}" == "prod" ]; then
    echo "🎉 Production deployment complete!"
else
    echo "✅ ${TARGET} deployment complete!"
fi
```

#### Example 3: Cluster Management Script

```bash
#!/bin/bash
# manage_cluster.sh

ACTION=$1
CLUSTER_ID=$2

case $ACTION in
  start)
    echo "▶️  Starting cluster ${CLUSTER_ID}..."
    databricks clusters start --cluster-id ${CLUSTER_ID}
    ;;
  stop)
    echo "⏸️  Stopping cluster ${CLUSTER_ID}..."
    databricks clusters stop --cluster-id ${CLUSTER_ID}
    ;;
  restart)
    echo "🔄 Restarting cluster ${CLUSTER_ID}..."
    databricks clusters restart --cluster-id ${CLUSTER_ID}
    ;;
  status)
    echo "📊 Cluster status:"
    databricks clusters get --cluster-id ${CLUSTER_ID}
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status} <cluster-id>"
    exit 1
    ;;
esac
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Pre-commit Hooks

This project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📄 License

This project is part of the Marvelous MLOps educational initiative. Please refer to the repository for specific license information.

## 🌟 Acknowledgments

- **Dataset**: Marvel Characters Dataset from Kaggle
- **Platform**: Databricks for serverless ML infrastructure
- **Community**: Open-source MLOps community for best practices

## 📞 Contact & Resources

- **Repository**: [marvelousmlops/marvel-characters](https://github.com/marvelousmlops/marvel-characters)
- **Organization**: [Marvelous MLOps](https://github.com/marvelousmlops)
- **Databricks Documentation**: [Serverless Documentation](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/three)
- **UV Documentation**: [UV Getting Started](https://docs.astral.sh/uv/getting-started/installation/)

## 🎓 Learning Resources

This project is designed as a learning resource for:
- Data Scientists transitioning to MLOps
- ML Engineers building production pipelines
- Students learning end-to-end ML workflows
- Teams implementing Databricks best practices

---

**Built with ❤️ by the Marvelous MLOps community**

*Data provided by Marvel. © Marvel*
