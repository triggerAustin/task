#!/bin/bash
# Runs the verifier inside the task's environment image. All dependencies are baked
# into environment/Dockerfile, so nothing is installed here.
#
# Grading plumbing is hardened against the agent, which controls /app and may have
# written to /logs during its run:
#   * PYTHONSAFEPATH stops Python prepending the working directory to sys.path, so
#     agent-written modules cannot shadow stdlib or plugin imports.
#   * the verifier runs from /tests with an explicit rootdir, so a conftest.py or
#     pytest.ini planted under /app is never collected.
#   * the reward directory is recreated from scratch, discarding any file, symlink
#     or directory the agent may have pre-placed there.
set -uo pipefail

export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONSTARTUP

rm -rf /logs/verifier
mkdir -p /logs/verifier

cd /tests || exit 1

python3 -m pytest /tests/test_outputs.py \
    --rootdir=/tests -c /dev/null \
    -v --tb=short -p no:cacheprovider \
    --ctrf /logs/verifier/ctrf.json
status=$?

if [ "$status" -eq 0 ]; then
    printf '1' > /logs/verifier/reward.txt
else
    printf '0' > /logs/verifier/reward.txt
fi

exit 0
