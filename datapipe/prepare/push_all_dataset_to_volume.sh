#!/bin/bash
#
# This script calls another script (e.g., split_across_volumes_file_move_folders.sh)
# multiple times with different arguments.
#
# Make sure to:
#  1) Adjust the path to the main script you want to run.
#  2) Fill in the RUNS array with your desired configurations.
#  3) Make this script executable: chmod +x run_multiple_splits.sh
#  4) Run it: ./run_multiple_splits.sh
set -o errexit
# Path to the main script you want to run multiple times:
chmod +x split_and_push.sh
MAIN_SCRIPT="./split_and_push.sh"

# If the script is located elsewhere, give the full or relative path, e.g.:
# MAIN_SCRIPT="/home/user/scripts/split_across_volumes_file_move_folders.sh"

# ----------------------------------------------------------------------------
# Define each run as a single string with fields separated by '|'.
# Format: "SOURCE_FOLDER|FILES_PER_SUBFOLDER|DESTINATION_FOLDER|VOLUME_FILE"
#
# For example, you might want to run the main script on:
#   - /data/source1 with 100 items per subfolder
#   - /data/source2 with 200 items per subfolder, etc.
# Adjust to suit your needs.
# ----------------------------------------------------------------------------
declare -a RUNS=(
  # "/data/FFHQ/Dataset/FFHQ/val|10000|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_FFHQ_val.txt|10"
  # "/data/FFHQ/Dataset/FFHQ/test|5000|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_FFHQ_test.txt|20"
  # "/data/FFHQ/Dataset/FFHQ/train|5000|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_FFHQ_train.txt|12"
  # "/data/FFHQ/Dataset/CelebA-HQ/train|5000|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_CelebAHQ_train.txt|5"
  # "/data/FFHQ/Dataset/CelebA-HQ/test|5000|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_CelebAHQ_test.txt|1"
  # "/data/FFHQ/Dataset/CelebA-HQ/val|5000|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_CelebAHQ_val.txt|1"
  # "/data/FFHQ/Dataset/Places2/test_256|4999|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_Places2_test.txt|20"
  # "/data/FFHQ/Dataset/Places2/val_256|4999|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_Places2_val.txt|20"
  # "/data/FFHQ/Dataset/Places2/train|4999|/data/FFHQ/Temp_dataset2|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_Places2_train.txt|20"
  # "/data/FFHQ/Dataset/ImageNet/val|4999|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_ImageNet_val.txt|20"
  # "/data/FFHQ/Dataset/ImageNet/test|4999|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_ImageNet_test.txt|20"
  # "/data/FFHQ/Dataset/ImageNet/train|4999|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_ImageNet_train.txt|20"
  # "/data/FFHQ/Dataset/Mask/test|4999|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_Mask_test.txt|20"
  # "/data/FFHQ/Dataset/Mask/val|4999|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_Mask_val.txt|20"
  "/data/FFHQ/Dataset/Mask/train|4999|/data/FFHQ/Temp_dataset|/data/FFHQ/DifFace_Thesis/datapipe/prepare/Vol_Mask_train.txt|20"
)




# Loop over each configuration and run the main script
for CONFIG in "${RUNS[@]}"; do
  IFS="|" read -ra ARGS <<< "$CONFIG"
  SRC="${ARGS[0]}"
  PER_SUB="${ARGS[1]}"
  DEST="${ARGS[2]}"
  VOLFILE="${ARGS[3]}"
  NUMPERVOL="${ARGS[4]}"

  echo "-------------------------------------------------------------"
  echo "Running: $MAIN_SCRIPT"
  echo "  SOURCE_FOLDER:         $SRC"
  echo "  FILES_PER_SUBFOLDER:   $PER_SUB"
  echo "  DESTINATION_FOLDER:    $DEST"
  echo "  VOLUME_FILE:           $VOLFILE"
  echo "  NUM_LOOP_PER_VOLUME:   $NUMPERVOL"
  echo 

  # Call the main script with the parsed arguments
  bash "$MAIN_SCRIPT" "$SRC" "$PER_SUB" "$DEST" "$VOLFILE" "$NUMPERVOL"

  echo "-------------------------------------------------------------"
  echo
done

echo "All runs completed."
