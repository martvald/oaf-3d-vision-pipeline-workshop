#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git-lfs libgl1 libglib2.0-0

git lfs install
unset GIT_LFS_SKIP_SMUDGE || true
git lfs pull

curl -fsSL https://static.pantsbuild.org/setup/get-pants.sh | bash
pants export --resolve=default

VENV_ACTIVATE="$(find dist/export/python/virtualenvs -name activate -type f | head -n1)"
echo "source ${VENV_ACTIVATE}" >> ~/.bashrc
source "${VENV_ACTIVATE}"

pip install --no-cache-dir jupyterlab
