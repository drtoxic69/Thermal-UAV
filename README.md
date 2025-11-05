# Thermal UAV Object Detection Framework

[](https://www.python.org/downloads/)
[](https://github.com/astral-sh/ruff)
[](https://github.com/microsoft/pyright)
[](https://www.google.com/search?q=https://github.com/drtoxic69/Thermal-UAV/actions/workflows/ci.yaml)

An AI system to detect humans and objects in thermal drone imagery using deep learning. This project leverages publicly available UAV thermal datasets and fine-tunes modern YOLO models to identify heat signatures even in low-visibility conditions such as night, fog, or smoke.

-----

## Project Goals

Based on our initial literature survey, this project aims to build a single, robust framework that addresses three key challenges in thermal object detection:

1.  **Nonuniformity Correction (NUC):** Implementing pre-processing algorithms to clean and enhance the raw thermal sensor data before it's fed to the model.
2.  **RGB-T Multimodal Fusion:** Developing a modified YOLO architecture (e.g., with attention mechanisms) that can effectively fuse features from both standard visual (RGB) and thermal (T) cameras for improved accuracy.
3.  **Small Object Detection:** Integrating an adaptive tiling strategy (e.g., SAHI) during inference to accurately detect small, distant objects, which are common in aerial UAV footage.

-----

## Getting Started

Follow these instructions to set up your local development environment.

### 1. Clone the Repository

```bash
git clone https://github.com/drtoxic69/Thermal-UAV.git
cd Thermal-UAV
```

### 2. Set Up the Environment

We use `uv` for package and environment management.

```bash
# Create the virtual environment
uv venv
# or
uv venv --prompt UAV --python 3.14

# Activate the environment
# On macOS/Linux (zsh/bash):
source .venv/bin/activate
# On Windows (powershell):
.venv\Scripts\activate
```

### 3. Install Dependencies

Install all project and development dependencies from the lock file.

```bash
uv sync
```

### 4. Set Up Environment Variables

Your local test suite (and other scripts) will read data paths from a `.env` file, which is ignored by Git.

```bash
# Create your local .env file from the template
cp .example.env .env

```

Now, **edit the `.env` file** and set `TEST_DATA_PATH` to the correct location of the VEDAI dataset on your machine.

```bash
# Source the .env file
source .env # bash/zsh

# or

./.env # powershell
```


### 5. Download the Data

Our `DataLoader` is built to handle multiple datasets. Download the starter data and place it in the `data/raw/` directory (which is in `.gitignore`).

  * **VEDAI:** Download the `512x512 images` and `annotations` from the [official site](https://downloads.greyc.fr/vedai/).
      * Unzip and place them so your structure looks like this:
        ```
        data/
        └── raw/
            └── vedai/
                ├── Annotations512/   (This is the original name)
                └── Vehicules512/
        ```
  * **Other Datasets:** (FLIR, M3FD, etc.) should also be placed in `data/raw/`.

-----

## Developer Workflow

This project is protected. You cannot push directly to `main`.

### 1. The Git Flow

1.  Always start from an up-to-date `main` branch:
    ```bash
    git checkout main
    git pull
    ```
2.  Create your new feature branch:
    ```bash
    git checkout -b feature/my-cool-feature
    ```
3.  Do your work and commit your changes.
4.  Push your branch to the remote:
    ```bash
    git push -u origin feature/my-cool-feature
    ```
5.  Open a Pull Request on GitHub to merge your branch into `main`.

## 2. The Golden Rule: Test Before You Push

Our CI pipeline will run all checks, but you **must** run them locally first. If these checks fail, your PR will be blocked.

```bash
# 1. Auto-format your code
uv run ruff format . --fix

# 2. Check for linting errors
uv run ruff check . --fix

# 3. Check for linting errors
uv run ruff check --select I --fix                                                  

# 4. Check for type-hinting errors
uv run pyright .

# 5. Run the full test suite
uv run pytest
```

Only `git push` after all four of these commands pass.

-----

## Usage

### Running Tests

To run all unit and integration tests:

```bash
uv run pytest
```

### Training

(Placeholder)

```bash
uv run python train.py --config configs/baseline_yolo.yaml
```

### Inference

(Placeholder)

```bash
uv run python inference.py --rgb-image path/to/img.png --thermal-image path/to/img_ir.png
```

-----

## Project Team

  * **Shivakumar:** Project Lead & Data Engineer
  * **Rayyan:** Pre-processing & NUC Specialist
  * **Sneha:** Core Model & Fusion Architect
  * **Yashas:** Post-processing & Evaluation Specialist

-----

## License

This project is licensed under the **MIT License**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
