from pathlib import Path

# NOTE: This is the root path of the Olive workspace in the container.
# It is used to mount the files to the container.
# The files are mounted to the container at the same path as the local files.
CONTAINER_ROOT_PATH = Path("/olive-ws/")
