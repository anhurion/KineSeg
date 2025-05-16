#!/bin/sh
set -e

VIDEO_PATH=""
VIDEO_PROVIDED=false
TEST_MODE=false
IT_MODE=false

NEW_ARGS=()
prev=""

# --video, --test, --it checks, remove them from NEW_ARGS
for arg in "$@"; do
  if [ "$prev" = "--video" ]; then
    VIDEO_PROVIDED=true
    VIDEO_PATH="$arg"
  fi

  if [ "$arg" = "--test" ]; then
    TEST_MODE=true
  elif [ "$arg" = "--it" ]; then
    IT_MODE=true
  elif [ "$arg" != "--video" ] && [ "$prev" != "--video" ]; then
    NEW_ARGS+=("$arg")
  fi

  prev="$arg"
done

if [ "$VIDEO_PROVIDED" = true ]; then
  # absolute video path on host
  HOST_VIDEO="$(realpath "$VIDEO_PATH")"
  # inside‐container target (just basename)
  BASE="$(basename "$VIDEO_PATH")"
  CONTAINER_VIDEO="/mnt/input/${BASE}"
  # bind‐mount the single file
  VIDEO_MOUNT="-v ${HOST_VIDEO}:${CONTAINER_VIDEO}:ro"
  # append rewritten --video argument
  NEW_ARGS+=(--video "${CONTAINER_VIDEO}")
else
  VIDEO_MOUNT=""
fi

# outputs folder
mkdir -p outputs
OUT_MOUNT="-v $(pwd)/outputs:/env/outputs"

# common flags
COMMON_FLAGS="--rm --gpus all --net=host ${OUT_MOUNT} ${VIDEO_MOUNT}"

# Modes
if [ "$TEST_MODE" = true ] && [ "$IT_MODE" = true ]; then
  echo ">>> Interactive Test mode"
  docker run ${COMMON_FLAGS} \
    -v "$(pwd)":/env/testing:rw \
    -it --entrypoint /bin/zsh \
    hmr_aris:v4_auto \
    -c "\
    cp -r /env/CameraHMR_IBRICS_fork/data /env/testing/data && \
    cd /env/testing && \
    exec /bin/zsh"

elif [ "$TEST_MODE" = true ]; then
  echo ">>> Test mode: copying data via entrypoint override"
  docker run ${COMMON_FLAGS} \
    -v "$(pwd)/CameraHMR_IBRICS_fork":/env/testing:rw \
    --entrypoint /bin/zsh \
    hmr_aris:v4_auto \
    -c "\
      cp -r /env/CameraHMR_IBRICS_fork/data /env/testing/data && \
      cd /env/testing && \
      exec python ibrics_main.py ${NEW_ARGS[*]} \
    "

else
  echo ">>> Normal mode"
  echo "${CONTAINER_VIDEO}"
  docker run ${COMMON_FLAGS} \
    hmr_aris:v4_auto \
    "${NEW_ARGS[@]}"
fi
