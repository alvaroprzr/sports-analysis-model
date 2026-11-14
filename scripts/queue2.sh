#!/bin/bash
# Resumable experiment queue. Each line of the step file is "name|command"; a step that finished
# leaves data/cache/queue/<name>.done and is skipped on the next run, so the queue can be stopped
# at any time and relaunched to continue where it was.
#
#   scripts/queue.sh STEP_FILE
cd "$(dirname "$0")/.."
STEPS=$1
mkdir -p data/cache/queue
while IFS='|' read -r name cmd; do
    [ -z "$name" ] || [[ "$name" == \#* ]] && continue
    [ -f "data/cache/queue/$name.done" ] && { echo "skip $name (done)"; continue; }
    # every step peaks at 3-4 GB of the 7 GB WSL has: never start one next to another
    while [ "$(free -g | awk '/^Mem:/ {print $7}')" -lt "${MIN_FREE_GB:-5}" ]; do
        echo "   waiting for memory before $name ($(free -g | awk '/^Mem:/ {print $7}') GB available)"
        sleep 60
    done
    echo "== $name $(date +%H:%M)"
    bash -c "$cmd" > "data/cache/queue/$name.log" 2>&1
    rc=$?
    echo "- [cola $(basename "$STEPS")] $name rc=$rc $(date "+%d/%m %H:%M")" >> data/cache/queue/history.log
    echo "   $name rc=$rc $(date +%H:%M)"
    [ $rc -eq 0 ] && touch "data/cache/queue/$name.done" || exit $rc
done < "$STEPS"
echo "QUEUE FINISHED $(date +%H:%M)"
