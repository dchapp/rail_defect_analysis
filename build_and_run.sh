#!/bin/bash
### Get cwd
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

### Build the container
docker build -t dchapp/rail_defect_analysis . 

### Run the container interactively with 
### local machine's src directory mirrored
### with container's src directory.
docker run -itv $DIR/src/:/src/ dchapp/rail_defect_analysis bash
