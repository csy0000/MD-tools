#!/usr/bin/env python3
"""Run the ALA campaign across the GPUs you name, launching each job as devices free up.

    python schedule.py --gpus 0,1,2,3,4,5,6,7,8 --no-ladder-gpus 0

Run from a campaign directory laid out as this one describes: `implicit.{pdb,xml}` and
`explicit.{pdb,xml}` at the top, and one `md-openmm build-md` output directory per job named as in
`JOBS`, each holding its generated `run.sh`.

Two rules the allocation obeys:
  * a job starts only when it can have every device it needs at once;
  * a REST2 ladder never gets a device named in `--no-ladder-gpus` -- on the machine this ran on,
    device 0 was a different card model, and a ladder spanning it would give one rung different
    hardware.

Longest jobs are queued first so the tail is short. The device list is an ARGUMENT, not a constant:
which devices exist, and which to keep a ladder off, is a fact about one machine.

`hot_explicit` is not in `JOBS`. It must start from `cold_explicit_r1`'s NPT-equilibrated box and is
launched by `run_from_equilibrated.sh` once that exists; see the README.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path.cwd()
POLL = 60

#: name, devices, topology stem, is a ladder, hours (for ordering only)
JOBS = [
    ("cold_explicit_r1", 1, "explicit", False, 13.0),
    ("cold_explicit_r2", 1, "explicit", False, 13.0),
    ("cold_explicit_r3", 1, "explicit", False, 13.0),
    ("cold_implicit_r1", 1, "implicit", False, 12.0),
    ("cold_implicit_r2", 1, "implicit", False, 12.0),
    ("cold_implicit_r3", 1, "implicit", False, 12.0),
    ("rest2_explicit", 6, "explicit", True, 8.0),
    ("rest2_implicit", 4, "implicit", True, 7.0),
    ("hot_implicit", 1, "implicit", False, 4.0),
]


def externally_busy():
    """Devices held by anything this scheduler did not start.

    Jobs launched by an earlier run of this script, or by hand, are invisible to `running` -- and
    the first version allocated straight on top of three of them. Ask the driver instead of
    trusting bookkeeping: a device with a compute process on it is not free, whoever put it there.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return set()
    uuids = {u.strip() for u in out.stdout.split() if u.strip()}
    index = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                           capture_output=True, text=True)
    busy = set()
    for line in index.stdout.splitlines():
        if "," in line:
            idx, uuid = (x.strip() for x in line.split(",", 1))
            if uuid in uuids:
                busy.add(int(idx))
    return busy


def note(msg):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with (HERE / "schedule.log").open("a") as fh:
        fh.write(line + "\n")


def launch(name, gpus, stem):
    d = HERE / name
    env_gpus = ",".join(str(g) for g in gpus)
    # No PYTHONPATH. `importlib.metadata` reads the INSTALLED distribution, so a run pointed at a
    # source tree records the installed version while running other code. The campaign did that
    # for 37 minutes and those runs were discarded.
    cmd = (f"cd {d} && CUDA_VISIBLE_DEVICES={env_gpus} setsid nohup ./run.sh "
           f"../{stem}.pdb ../{stem}.xml > campaign.log 2>&1 &")
    subprocess.run(["bash", "-lc", cmd], check=True)
    note(f"START {name} on GPU {env_gpus}")


def alive(name):
    """Is this job still running?

    Matching `f"{name}/"` did not work and is why the first run of this scheduler declared the
    whole campaign complete in two minutes: `run.sh` is launched as `./run.sh ../x.pdb ../x.xml`
    from inside its own directory, so the job name appears nowhere in its command line. The
    launcher wrapper DOES carry the absolute path, so match that.
    """
    out = subprocess.run(["pgrep", "-f", str(HERE / name)], capture_output=True, text=True)
    return bool(out.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpus", required=True,
                        help="comma-separated device indices this campaign may use")
    parser.add_argument("--no-ladder-gpus", default="",
                        help="comma-separated devices a REST2 ladder must not be given")
    parser.add_argument("--jobs", nargs="*", help="run only these jobs (default: all of JOBS)")
    args = parser.parse_args()
    all_gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    no_ladder = {int(g) for g in args.no_ladder_gpus.split(",") if g.strip()}

    pending = sorted((j for j in JOBS if not args.jobs or j[0] in args.jobs),
                     key=lambda j: -j[4])
    running = {}                                   # name -> gpus
    state = HERE / "schedule.state.json"
    note(f"campaign start: {len(pending)} job(s) queued on GPU {args.gpus}")
    while pending or running:
        for name in list(running):
            if not alive(name):
                note(f"DONE  {name} (released GPU {','.join(map(str, running[name]))})")
                del running[name]
        busy = {g for gs in running.values() for g in gs} | externally_busy()
        free = [g for g in all_gpus if g not in busy]
        for job in list(pending):
            name, n, stem, is_ladder, _ = job
            usable = [g for g in free if not (is_ladder and g in no_ladder)]
            if len(usable) >= n:
                gpus = usable[:n]
                launch(name, gpus, stem)
                running[name] = gpus
                free = [g for g in free if g not in gpus]
                pending.remove(job)
        state.write_text(json.dumps({"running": running, "pending": [j[0] for j in pending]},
                                    indent=2))
        if pending or running:
            time.sleep(POLL)
    note("campaign complete")


if __name__ == "__main__":
    main()
