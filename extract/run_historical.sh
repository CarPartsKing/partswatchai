#!/bin/bash
START=${1:-0}
END=${2:-197}
LOG=/tmp/historical_load.log

echo "$(date) Starting historical load from chunk $START to $END" >> "$LOG"

for i in $(seq $START $((END-1))); do
    echo "$(date) === CHUNK $i ===" >> "$LOG"
    python -m extract.autocube_pull --mode historical --start-chunk $i --max-chunks 1 >> "$LOG" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "$(date) Chunk $i exited with code $RC" >> "$LOG"
    fi
    echo "$(date) Chunk $i done (exit=$RC)" >> "$LOG"
done

echo "$(date) Historical load complete" >> "$LOG"
