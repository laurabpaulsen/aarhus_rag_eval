#!/usr/bin/env bash


scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
echo $scriptDir
python3.10 -m venv "$scriptDir/env"
source "$scriptDir/env/bin/activate"
python3.10 -m pip install -r "$scriptDir/requirements.txt"
