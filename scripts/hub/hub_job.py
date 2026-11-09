"""Run long jobs on the team JupyterHub from this PC.

Usage:
    python scripts/hub/hub_job.py upload LOCAL_FILE REMOTE_PATH
    python scripts/hub/hub_job.py start REMOTE_SCRIPT LOG_NAME [ARGS...]
    python scripts/hub/hub_job.py log LOG_NAME [N_LINES]
    python scripts/hub/hub_job.py download REMOTE_PATH LOCAL_FILE
    python scripts/hub/hub_job.py run LOCAL_CODE_FILE

Remote paths are relative to the hub home folder. `start` launches the script with nohup inside
the project folder on the hub, so it keeps running if this PC disconnects; logs go to
~/ift7030-lacha/logs/. The token is read from .secrets/jupyterhub_token.
"""

import base64
import json
import sys
import uuid
from pathlib import Path

import requests
import websocket

HUB = "https://jupyter.ice.ulaval.ca"
SERVER = HUB + "/user/lacha188"
PROJECT_ON_HUB = "ift7030-lacha"
TOKEN = (Path(__file__).resolve().parents[2] / ".secrets/jupyterhub_token").read_text().strip()

session = requests.Session()
session.headers["Authorization"] = "token " + TOKEN


def run_code(code, timeout=600):
    """Execute Python code in a temporary kernel and print its output."""
    kernel_id = session.post(SERVER + "/api/kernels", json={"name": "python3"}, timeout=60).json()["id"]
    failed = False
    try:
        connection = websocket.create_connection(
            SERVER.replace("https:", "wss:") + f"/api/kernels/{kernel_id}/channels",
            header=["Authorization: token " + TOKEN],
            timeout=timeout,
        )
        message_id = uuid.uuid4().hex
        connection.send(
            json.dumps(
                {
                    "header": {
                        "msg_id": message_id,
                        "msg_type": "execute_request",
                        "username": "agent",
                        "session": uuid.uuid4().hex,
                        "version": "5.3",
                    },
                    "parent_header": {},
                    "metadata": {},
                    "channel": "shell",
                    "content": {"code": code, "silent": False, "store_history": False, "allow_stdin": False},
                }
            )
        )
        while True:
            message = json.loads(connection.recv())
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "stream":
                print(content["text"], end="", flush=True)
            elif kind == "error":
                failed = True
                print(content["ename"], content["evalue"], flush=True)
            elif kind == "status" and content["execution_state"] == "idle":
                break
        connection.close()
    finally:
        session.delete(SERVER + f"/api/kernels/{kernel_id}", timeout=30)
    if failed:
        raise SystemExit(1)


def upload(local_file, remote_path):
    parts = remote_path.split("/")[:-1]
    for depth in range(1, len(parts) + 1):
        folder = "/".join(parts[:depth])
        if session.get(SERVER + "/api/contents/" + folder, params={"content": 0}, timeout=30).status_code == 404:
            session.put(SERVER + "/api/contents/" + folder, json={"type": "directory"}, timeout=30).raise_for_status()
    content = base64.b64encode(Path(local_file).read_bytes()).decode()
    response = session.put(
        SERVER + "/api/contents/" + remote_path,
        json={"type": "file", "format": "base64", "content": content},
        timeout=300,
    )
    response.raise_for_status()
    print("uploaded", local_file, "->", remote_path)


def download(remote_path, local_file):
    response = session.get(
        SERVER + "/api/contents/" + remote_path, params={"content": 1, "format": "base64"}, timeout=300
    )
    response.raise_for_status()
    Path(local_file).parent.mkdir(parents=True, exist_ok=True)
    Path(local_file).write_bytes(base64.b64decode(response.json()["content"]))
    print("downloaded", remote_path, "->", local_file)


def start(remote_script, log_name, args):
    run_code(
        f"""
import subprocess, pathlib
project = pathlib.Path.home() / {PROJECT_ON_HUB!r}
(project / "logs").mkdir(exist_ok=True)
log = open(project / "logs" / {log_name!r}, "w")
process = subprocess.Popen(
    ["python", "-u", str(pathlib.Path.home() / {remote_script!r}), *{args!r}],
    cwd=project, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
)
print("started pid", process.pid)
"""
    )


def show_log(log_name, n_lines):
    run_code(
        f"""
import pathlib, subprocess
path = pathlib.Path.home() / {PROJECT_ON_HUB!r} / "logs" / {log_name!r}
print("".join(path.read_text().splitlines(keepends=True)[-{n_lines}:]))
print(subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout)
"""
    )


if __name__ == "__main__":
    command, rest = sys.argv[1], sys.argv[2:]
    if command == "upload":
        upload(*rest)
    elif command == "download":
        download(*rest)
    elif command == "start":
        start(rest[0], rest[1], rest[2:])
    elif command == "log":
        show_log(rest[0], int(rest[1]) if len(rest) > 1 else 30)
    elif command == "run":
        run_code(Path(rest[0]).read_text())
    else:
        raise SystemExit(__doc__)
