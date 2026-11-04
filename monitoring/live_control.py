"""Inspect the current live experiment and issue explicit anomaly commands."""

import argparse
import json
from pathlib import Path
import time
import uuid

from monitoring.psi import require, write_json

POINTER = Path("reports/psi_live_pointer.json")


def current():
    value = json.loads(POINTER.read_text())
    folder = Path(value["folder"])
    require(folder.resolve().is_relative_to((Path.cwd() / "reports").resolve()), "Experiment evidence is outside reports")
    return value, folder


def write_control(path, anomaly, source):
    require(anomaly in ("none", "combined"), "E1 supports none or combined")
    value = {"schema_version": 1, "request_id": uuid.uuid4().hex,
             "anomaly": anomaly, "source": source, "requested_unix": time.time()}
    write_json(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--container", required=True)
    register.add_argument("--folder", type=Path, required=True)
    sub.add_parser("status")
    sub.add_parser("container-name")
    request = sub.add_parser("request")
    request.add_argument("--anomaly", choices=("none", "combined"), required=True)
    wait = sub.add_parser("wait")
    wait.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.command == "register":
        write_json(POINTER, {"container": args.container, "folder": str(args.folder), "registered_unix": time.time()})
        return 0
    value, folder = current()
    if args.command == "container-name":
        print(value["container"])
        return 0
    if args.command == "request":
        status = json.loads((folder / "live_status.json").read_text())
        require(status["status"] == "running", "The experiment is not running")
        print(json.dumps(write_control(folder / "control.json", args.anomaly, "manual_cli"), indent=2))
        return 0
    if args.command == "wait":
        deadline = time.monotonic() + args.timeout
        while True:
            path = folder / "live_status.json"
            if path.exists():
                status = json.loads(path.read_text())
                if status["status"] in ("running", "completed"):
                    print("Live MQTT experiment ready; container:", value["container"])
                    print("Evidence folder:", folder)
                    print("Run ID:", status["run_id"])
                    return 0
                require(status["status"] not in ("failed", "interrupted"), "Live startup failed: " + str(status.get("error")))
            require(time.monotonic() < deadline, "Live startup did not become ready within 30 seconds")
            time.sleep(.25)
    path = folder / "live_status.json"
    require(path.is_file(), "Status is not available yet; inspect the experiment container logs")
    print(json.dumps({"container": value["container"], "evidence_folder": str(folder),
                      **json.loads(path.read_text())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
