"""Run scripts one after the other on the hub, after an already running process ends.

    python scripts/hub/run_queue.py WAIT_PID "script.py --quick" "script.py" ...

Each step runs from the project folder; its output goes to logs/queue_<n>.log. The queue stops at
the first step that fails, so a broken quick test never launches the long run.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

wait_pid = int(sys.argv[1])
while Path(f"/proc/{wait_pid}").exists():
    time.sleep(60)

for number, step in enumerate(sys.argv[2:], start=1):
    log = Path("logs") / f"queue_{number}.log"
    print(f"{time.strftime('%H:%M')} step {number}: {step} -> {log}", flush=True)
    with open(log, "w") as out:
        result = subprocess.run(["python", "-u", *step.split()], stdout=out, stderr=subprocess.STDOUT, env=os.environ)
    print(f"{time.strftime('%H:%M')} step {number} exit code {result.returncode}", flush=True)
    if result.returncode != 0:
        break
print("QUEUE DONE", flush=True)
