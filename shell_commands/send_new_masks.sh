#!/bin/bash

# Local and remote base directories
LOCAL_BASE_DIR="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/patches_xenium"
REMOTE_BASE_DIR="/gpfs/workdir/user/HE2CellType/CT_DS/check_align_patches/apply_cellvit/prepared_patches_xenium"
REMOTE_USER="user"
REMOTE_HOST="ruche.mesocentre.universite-paris-saclay.fr"

# List of slide IDs (example)
SLIDE_IDS=(breast_s0 breast_s1 breast_s3 breast_s6 lung_s1 lung_s3 skin_s1 skin_s2 skin_s3 skin_s4 pancreatic_s0 pancreatic_s1 pancreatic_s2 heart_s0 colon_s1 colon_s2 kidney_s0 kidney_s1 liver_s0 liver_s1 tonsil_s0 tonsil_s1 lymph_node_s0 ovary_s0 ovary_s1 brain_s0 bone_marrow_s0 bone_marrow_s1 bone_s0 prostate_s0 cervix_s0)

# Iterate over each slide ID
for SLIDE_ID in "${SLIDE_IDS[@]}"; do
  LOCAL_FILE="${LOCAL_BASE_DIR}/${SLIDE_ID}/patch_ids.npy"
  REMOTE_DIR="${REMOTE_BASE_DIR}/${SLIDE_ID}/fold2/"

  if [ -f "${LOCAL_FILE}" ]; then
    echo "Transferring ${LOCAL_FILE} to ${REMOTE_DIR}..."
    rsync --partial --progress -r "${LOCAL_FILE}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
  else
    echo "File ${LOCAL_FILE} does not exist. Skipping..."
  fi
done