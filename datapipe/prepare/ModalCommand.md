Link: https://modal.com/docs/reference/cli/volume

create: Create a named, persistent modal.Volume.
modal volume create [OPTIONS] NAME

get: Download files from a modal.Volume object.
modal volume get [OPTIONS] VOLUME_NAME REMOTE_PATH [LOCAL_DESTINATION]

list: List the details of all modal.Volume volumes in an Environment.
modal volume list [OPTIONS]

ls: List files and directories in a modal.Volume volume.
modal volume ls [OPTIONS] VOLUME_NAME [PATH]

put: Upload a file or directory to a modal.Volume.
modal volume put [OPTIONS] VOLUME_NAME LOCAL_PATH [REMOTE_PATH]

rm: Delete a file or directory from a modal.Volume.
modal volume rm [OPTIONS] VOLUME_NAME REMOTE_PATH

cp: Copy within a modal.Volume.
modal volume cp [OPTIONS] VOLUME_NAME PATHS...

delete: Delete a named, persistent modal.Volume.
modal volume delete [OPTIONS] VOLUME_NAME

rename: Rename a modal.Volume.
modal volume rename [OPTIONS] OLD_NAME NEW_NAME