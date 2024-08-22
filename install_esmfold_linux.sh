#!/bin/bash -e

# Check if wget is installed, if not, exit with a message
type wget 2>/dev/null || { echo "wget is not installed. Please install it using apt or yum." ; exit 1 ; }

# Function to generate a random string of a given length
generate_random_string() {
  local length=$1
  if [ -z "$length" ]; then
    length=8  # Default length if not provided
  fi
  # Generate a random alphanumeric string using openssl
  openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c"$length"
}

# Generate a random output folder name
random_output_folder=$(generate_random_string 10)
OUTPUT_FOLDER="output_${random_output_folder}"

# Navigate to the scratch directory and create a temporary directory
cd $SCRATCHDIR
mkdir -p tmp
export TMPDIR="${SCRATCHDIR}/tmp"
export PIP_CACHE_DIR="${SCRATCHDIR}/tmp"

CURRENTPATH=$SCRATCHDIR
COLABFOLDDIR="${CURRENTPATH}/localcolabfold"

# Create the ColabFold directory and navigate into it
mkdir -p "${COLABFOLDDIR}"
cd "${COLABFOLDDIR}"

# Download and install Mambaforge (a minimal conda installation)
wget -q -P . https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash ./Mambaforge-Linux-x86_64.sh -b -p "${COLABFOLDDIR}/conda"
rm Mambaforge-Linux-x86_64.sh

# Source the conda environment and update PATH
source "${COLABFOLDDIR}/conda/etc/profile.d/conda.sh"
export PATH="${COLABFOLDDIR}/conda/condabin:${PATH}"

# Update conda and activate the base environment
conda update -n base conda -y
conda activate base

# Install necessary packages using pip
"$COLABFOLDDIR/conda/bin/pip" install --upgrade pip
"$COLABFOLDDIR/conda/bin/pip" install nvidia-cuda-runtime-cu12
"$COLABFOLDDIR/conda/bin/pip" install numpy scikit-learn pillow click protobuf
"$COLABFOLDDIR/conda/bin/pip" install scipy
"$COLABFOLDDIR/conda/bin/pip" install matplotlib
"$COLABFOLDDIR/conda/bin/pip" install biopython
"$COLABFOLDDIR/conda/bin/pip" install torch
"$COLABFOLDDIR/conda/bin/pip" install transformers accelerate

# Update PATH to include the colabfold-conda directory
export PATH="${COLABFOLDDIR}/colabfold-conda/bin:${PATH}"

# Create the input directory in the scratch directory and navigate to it
mkdir $SCRATCHDIR/input
cd $SCRATCHDIR/input

# Change directory to your output files directory and set write permissions
cd /path/to/output_files_directory/
chmod a+w /path/to/output_files_directory/

# Print the output folder name
echo "${OUTPUT_FOLDER}"

# Create the output folder and set write permissions
mkdir -p "${OUTPUT_FOLDER}"
chmod a+w "${OUTPUT_FOLDER}"

# Navigate back to the input directory
cd $SCRATCHDIR/input

# Run the Python script for ESMFold
python /path/to/esmfold_transformers.py
