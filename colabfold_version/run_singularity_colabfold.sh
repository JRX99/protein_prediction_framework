#!/bin/bash
# define variables
SING_IMAGE="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:23.11-py3.SIF"
HOMEDIR=/storage/plzen4-ntis/home/jrx99 # substitute username and path to to your real username and path

#set SINGULARITY variables for runtime data
export SINGULARITY_CACHEDIR=$HOMEDIR
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR

singularity exec --bind /storage/ \
$SING_IMAGE /storage/plzen4-ntis/home/jrx99/install_colabbatch_linux_2024.sh
