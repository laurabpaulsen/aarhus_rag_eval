#!/usr/bin/env bash


scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
echo $scriptDir
python3 -m venv "$scriptDir/env"
source "$scriptDir/env/bin/activate"
python3 -m pip install -r "$scriptDir/requirements.txt"
