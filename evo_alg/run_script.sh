echo "Installing packages......"
CURRENTPATH=$SCRATCHDIR

pip install --upgrade pip
pip install nvidia-cuda-runtime-cu12
pip install numpy scikit-learn pillow click protobuf
pip install scipy
pip install matplotlib
pip install biopython
pip install torch
pip install transformers accelerate
pip install seaborn

export TMPDIR=$SCRATCHDIR
echo "Creating folders......"
cd $SCRATCHDIR
mkdir input
mkdir output
mkdir results
export TMPDIR=$SCRATCHDIR/tmp

cd input
mkdir pdb_template
mkdir tmp

echo "Copying protein template......"
cp -r /path/to/source/pdb_template .

echo "Running script......"
python /path/to/script/evolutionary_algorithm.py

echo "Copying output files......"
cp -r $SCRATCHDIR/results /path/to/destination/outputs

echo "Deleting Scratch"
#rm -r $SCRATCHDIR/input

echo "Done......"
