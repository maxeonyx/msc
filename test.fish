. env/bin/activate.fish

set PREV_BRANCH_NAME (git symbolic-ref --short HEAD)
set -x RUN_NAME randomname get

# create a new commit with the current state of the working directory on a new branch
set STASH_NAME (git stash create)
cd ../msc-hands-latest-experiment-src
git checkout PREV_BRANCH_NAME
git stash apply $STASH_NAME
git checkout -b run-$RUN_NAME
git commit -A -m "Commit of src at run $RUN_NAME"
