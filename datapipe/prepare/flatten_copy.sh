#!/bin/bash
#
# flatten_copy.sh
#
# Usage:
#   ./flatten_copy.sh /path/to/source_folder /path/to/target_folder
#
# Description:
#   - Recursively find all .jpg files under "source_folder".
#   - Copy them into "target_folder" with NO subfolders (i.e., flatten).
#   - Convert subdirectory paths into filename prefixes. For example:
#       /train_256/a/airfield/indoor/00000006.jpg
#     becomes
#       a_airfield_indoor_00000006.jpg
#   - If the resulting filename already exists in the target folder,
#     append _1, _2, ... until a unique name is found.
#   - This preserves (copies) the source data; it does not delete or move them.

# 1) Check that we have exactly 2 arguments
if [ $# -ne 2 ]; then
  echo "Usage: $0 <SOURCE_FOLDER> <TARGET_FOLDER>"
  exit 1
fi

SOURCE_FOLDER="$1"
TARGET_FOLDER="$2"

# 2) Validate the source folder
if [ ! -d "$SOURCE_FOLDER" ]; then
  echo "Error: '$SOURCE_FOLDER' does not exist or is not a directory."
  exit 1
fi

# 3) Create the target folder if it doesn’t exist
mkdir -p "$TARGET_FOLDER"

# 4) Find and copy .jpg files
#    Flatten them by renaming subdirectory paths into filename prefixes.
#    Handle collisions by adding a numeric suffix.
find "$SOURCE_FOLDER" -type f -name "*.png" | while read -r FILEPATH; do
  # Get the base filename (e.g., "00000006.jpg")
  BASENAME="$(basename "$FILEPATH")"

  # Get the path relative to SOURCE_FOLDER
  #   e.g. if FILEPATH is /train_256/a/airfield/indoor/00000006.jpg
  #   and SOURCE_FOLDER is /train_256,
  #   then RELATIVE_PATH is a/airfield/indoor/00000006.jpg
  RELATIVE_PATH="${FILEPATH#$SOURCE_FOLDER/}"

  # Extract the directory part (e.g. "a/airfield/indoor") from that relative path
  RELATIVE_DIR="$(dirname "$RELATIVE_PATH")"

  # Turn all subfolder slashes into underscores, e.g. "a_airfield_indoor"
  PREFIX="$(echo "$RELATIVE_DIR" | tr / _)"

  # Construct the new flattened filename, e.g. "a_airfield_indoor_00000006.jpg"
  NEWNAME="${PREFIX}_${BASENAME}"

  # If there is no directory part (i.e., the file is directly in SOURCE_FOLDER),
  # then PREFIX might be "." — so let's handle that case:
  if [ "$PREFIX" = "." ]; then
    NEWNAME="$BASENAME"
  fi

  # Check if that filename already exists in the target folder
  if [ -e "$TARGET_FOLDER/$NEWNAME" ]; then
    # If so, loop to find a unique suffix, e.g. a_airfield_indoor_00000006_1.jpg
    SUFFIX=1
    EXT="${NEWNAME##*.}"        # file extension (e.g. "jpg")
    NAME="${NEWNAME%.*}"        # filename without extension (e.g. "a_airfield_indoor_00000006")
    while [ -e "$TARGET_FOLDER/${NAME}_$SUFFIX.$EXT" ]; do
      SUFFIX=$((SUFFIX + 1))
    done
    NEWNAME="${NAME}_$SUFFIX.$EXT"
  fi

  # Copy the file with the new flattened name
  # -p preserves timestamps and some attributes
  cp -p "$FILEPATH" "$TARGET_FOLDER/$NEWNAME"
done

echo "All .jpg files have been flattened and copied to '$TARGET_FOLDER'."
exit 0
