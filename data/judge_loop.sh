#!/bin/bash
# Re-judge new batches every 5 minutes while generators run; stop when none are left running.
cd "$(dirname "$0")"
while true; do
  python3 judge.py --jobs 6 || { echo "judge stopped (quota?)"; exit 2; }
  pgrep -f "gen.py" >/dev/null || { python3 judge.py --jobs 6; echo "generators done"; exit 0; }
  sleep 300
done
