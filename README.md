# Protein Interaction and Inhibitor Prediction Framework

![Python](https://img.shields.io/badge/python-3.x-blue.svg)

## Project Overview

This repository contains the code and resources for a computational framework designed to predict protein interactions and identify potential inhibitors. Utilizing state-of-the-art neural network model ESMFold, this framework enables the prediction of protein structures and interactions, as well as the design of peptides that can effectively inhibit target proteins.

### Key Features
- **Protein Structure Prediction:** Leverage ESMFold model for accurate and efficient protein structure predictions.
- **Inhibitor Design:** Identify and optimize peptide inhibitors through evolutionary algorithms.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)


## Installation

### Prerequisites
- Python 3.x
- Required Python libraries: `numpy`, `pandas`, `tensorflow`, `torch`, `biopython`, `matplotlib`
- Ensure you have access to Meta AI’s ESMFold model.

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    bash install_esmfold_linux.sh
    ```

4. Set up any additional dependencies or environment variables as required by your project.

## Usage

### Running the Prediction Framework

1. Prepare the protein sequence input files and configure the desired settings for the model in the configuration file.
2. Run the script to predict protein structures:
    ```bash
    python esmfold_transformers.py
    ```

3. Execute the evolutionary algorithm to identify potential inhibitors:
    ```bash
    python evolutionary_algorithm.py 
    ```

