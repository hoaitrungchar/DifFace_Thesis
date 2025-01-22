#!/bin/bash
#
# Usage:
#   ./split_across_volumes_file_copy_folders.sh /path/to/source_folder 100 /path/to/destination_folder /path/to/volume_list.txt
#
# Behavior:
#   1) Reads a list of volumes from volume_list.txt (one name per line).
#   2) Each volume can receive up to (5 * FILES_PER_SUBFOLDER) "items" at a time.
#      An "item" can be either a file or an entire folder.
#   3) After gathering the required number of items, the script:
#       a) Calls: modal put <volume_name> <subfolder_path>
#       b) Deletes that subfolder
#       c) Moves on to the next volume
#   4) Stops if it runs out of volumes or finishes all items.
#
#   Since this version uses 'cp -r', the source folder keeps its original items.

if [ $# -ne 5 ]; then
  echo "Usage: $0 <source_folder> <files_per_subfolder> <destination_folder> <volume_file> <num_loop_volume>"
  exit 1
fi

SOURCE_FOLDER=$1
FILES_PER_SUBFOLDER=$2
DESTINATION_FOLDER=$3
VOLUME_FILE=$4
NUM_PER_VOLUME=$5

# Validate source folder
if [ ! -d "$SOURCE_FOLDER" ]; then
  echo "Error: '$SOURCE_FOLDER' does not exist or is not a directory."
  exit 1
fi

# Validate files-per-subfolder (must be integer)
if ! [[ "$FILES_PER_SUBFOLDER" =~ ^[0-9]+$ ]]; then
  echo "Error: '$FILES_PER_SUBFOLDER' is not a valid number."
  exit 1
fi

# Validate volume file
if [ ! -f "$VOLUME_FILE" ]; then
  echo "Error: Volume file '$VOLUME_FILE' does not exist."
  exit 1
fi

# Create destination folder if needed
mkdir -p "$DESTINATION_FOLDER"

# Read the volume names into an array
mapfile -t VOLUMES < "$VOLUME_FILE"
volume_count=${#VOLUMES[@]}

if [ $volume_count -eq 0 ]; then
  echo "Error: No volume names found in $VOLUME_FILE."
  exit 1
fi

MAX_ITEMS_PER_VOLUME=$((NUM_PER_VOLUME * FILES_PER_SUBFOLDER))

# Counters
batch_number=0
current_batch_count=0
volume_index=0
current_loop=$5

create_subfolder() {
  batch_number=$((batch_number + 1))
  SUBFOLDER_NAME="sub_${batch_number}"
  mkdir -p "$DESTINATION_FOLDER/$SUBFOLDER_NAME"
}

finalize_subfolder() {
  local vol_name="$1"
  local folder_path="$2"

  echo "---------------------------------"
  echo "Batch complete for volume: $vol_name"
  echo "Contains $current_batch_count items: $folder_path"
  echo "Running: modal volume put $vol_name $folder_path"
  modal volume put "$vol_name" "$folder_path"

  echo "Deleting folder: $folder_path"
  rm -rf "$folder_path"
  echo
}

# Start with the first volume

current_volume="${VOLUMES[$volume_index]}"
create_subfolder

echo "$volume_count"

# Main loop: consider both files and directories as items
for item in "$SOURCE_FOLDER"/*; do
  if [ -d "$item" ] || [ -f "$item" ]; then
    # Copy entire folder or file; treat each as a single item
  # if [ "$batch_number" -eq 43 ] || [ "$batch_number" -eq 44 ] || [ "$batch_number" -eq 45 ] ||  [ "$batch_number" -eq 59 ] || [ "$batch_number" -eq 60 ]; then
    cp -r "$item" "$DESTINATION_FOLDER/$SUBFOLDER_NAME/"
  # fi

    ((current_batch_count++))

    if [ $current_batch_count -eq "$FILES_PER_SUBFOLDER" ]; then
      echo "$current_loop"
      echo "$volume_count"
      # if [ "$batch_number" -eq 43 ] || [ "$batch_number" -eq 44 ] || [ "$batch_number" -eq 45 ] ||  [ "$batch_number" -eq 59 ] || [ "$batch_number" -eq 60 ]; then
        finalize_subfolder "$current_volume" "$DESTINATION_FOLDER/$SUBFOLDER_NAME"
      # fi
      ((current_loop--))
      
      if [ $current_loop -le 0 ]; then
        ((volume_index++))
        current_loop=$5
        if [ $volume_index -ge $volume_count ]; then
          echo "No more volumes left. Stopping here."
        fi
        current_volume="${VOLUMES[$volume_index]}"
        
      fi
      create_subfolder
      current_batch_count=0


    fi
  fi
done

# Finalize if there's a partial batch
if [ $current_batch_count -gt 0 ]; then
  finalize_subfolder "$current_volume" "$DESTINATION_FOLDER/$SUBFOLDER_NAME"
fi

echo "All items processed (or no more volumes)."
exit 0
