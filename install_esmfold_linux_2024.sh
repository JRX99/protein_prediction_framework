#!/bin/bash -e

type wget 2>/dev/null || { echo "wget is not installed. Please install it using apt or yum." ; exit 1 ; }

generate_random_string() {
  local length=$1
  if [ -z "$length" ]; then
    length=8
  fi
  openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c"$length"
}

random_output_folder=$(generate_random_string 10)
OUTPUT_FOLDER="output_${random_output_folder}"

cd $SCRATCHDIR
mkdir -p tmp
export TMPDIR="${SCRATCHDIR}/tmp"
export PIP_CACHE_DIR="${SCRATCHDIR}/tmp"

CURRENTPATH=$SCRATCHDIR
COLABFOLDDIR="${CURRENTPATH}/localcolabfold"

mkdir -p "${COLABFOLDDIR}"
cd "${COLABFOLDDIR}"
wget -q -P . https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash ./Mambaforge-Linux-x86_64.sh -b -p "${COLABFOLDDIR}/conda"
rm Mambaforge-Linux-x86_64.sh

source "${COLABFOLDDIR}/conda/etc/profile.d/conda.sh"
export PATH="${COLABFOLDDIR}/conda/condabin:${PATH}"


conda update -n base conda -y
conda activate base

"$COLABFOLDDIR/conda/bin/pip" install --upgrade pip
"$COLABFOLDDIR/conda/bin/pip" install nvidia-cuda-runtime-cu12
"$COLABFOLDDIR/conda/bin/pip" install numpy scikit-learn pillow click protobuf
"$COLABFOLDDIR/conda/bin/pip" install scipy
"$COLABFOLDDIR/conda/bin/pip" install matplotlib
"$COLABFOLDDIR/conda/bin/pip" install biopython
"$COLABFOLDDIR/conda/bin/pip" install torch
"$COLABFOLDDIR/conda/bin/pip" install transformers accelerate

export PATH="${COLABFOLDDIR}/colabfold-conda/bin:${PATH}"
mkdir $SCRATCHDIR/input
cd $SCRATCHDIR/input
cd /storage/plzen4-ntis/home/jrx99/diplomka/output_files/
chmod a+w /storage/plzen4-ntis/home/jrx99/diplomka/output_files/
echo "${OUTPUT_FOLDER}"
mkdir -p "${OUTPUT_FOLDER}"
chmod a+w "${OUTPUT_FOLDER}"
cd $SCRATCHDIR/input
python /storage/plzen4-ntis/home/jrx99/esmfold_transformers.py











