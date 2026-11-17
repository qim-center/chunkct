import ast
import datetime
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import cloudpickle as pickle

from chunkct.recon import ChunkedCT


class DTULSFChunkedCT(ChunkedCT):
    """
    Submit ChunkedCT projection-writing and reconstruction work as DTU LSF jobs.

    The configured processor is serialized once per run and loaded by generated
    batch scripts, with reconstruction split into independent chunk jobs after
    projection writing. Uses cloudpickle so notebook-defined ProjectionSources
    and other on-the-fly definitions can be submitted without moving them to a
    separate module.
    """

    WRITE_QUEUE = "hpc"
    RECON_QUEUE = "gpuv100"
    RECON_GPU_SELECT = "gpu32gb"
    # Environment modules loaded in GPU jobs. Empty by default since the pixi/conda
    # environment ships its own CUDA runtime (only the driver is needed from the node).
    GPU_MODULES: tuple[str, ...] = ()

    ACTIVE_STATUSES = frozenset({"PEND", "RUN", "PSUSP", "USUSP", "SSUSP", "WAIT"})

    BENCHMARK_PROJ_SHAPE = (4713, 3000, 3000)
    BENCHMARK_RECON_CHUNK_SHAPE = (1500, 1500, 1500)
    BENCHMARK_WRITE_MINUTES = 20
    BENCHMARK_RECON_MINUTES = 15
    BENCHMARK_RECON_MEMORY_GB = 14

    # Extra resources if padding is used
    # Note his is an overestimate (especially for large padding) to be on the safe side
    PAD_FACTOR_WALLTIME_SLOPE = (1.75 - 1) / 0.25
    PAD_FACTOR_MEMORY_SLOPE = (1.5 - 1) / 0.25

    STATUS_DELIMITER = "|"
    STATUS_COMMAND = (
        'bjobs -a -o "'
        "jobid "
        "user "
        "queue "
        "job_name "
        "stat "
        "submit_time "
        "start_time "
        "run_time "
        "finish_time "
        "dependency "
        f"delimiter='{STATUS_DELIMITER}'"
        '"'
    )

    def __init__(
        self,
        projection_source,
        output_dir,
        angle_chunks,
        recon_chunk_shape=(1500, 1500, 1500),
        vol_chunk_shape=(100, 100, 100),
        verbose=1,
        pad_factor=None,
        filter="ram-lak",
    ):
        super().__init__(
            projection_source=projection_source,
            output_dir=output_dir,
            angle_chunks=angle_chunks,
            recon_chunk_shape=recon_chunk_shape,
            vol_chunk_shape=vol_chunk_shape,
            verbose=verbose,
            pad_factor=pad_factor,
            filter=filter,
        )
        # Jobs run with the interpreter of the submitting process (pixi/conda env),
        # so the package and any notebook-defined classes resolve identically.
        self.python_executable = sys.executable

    def _new_run_dir(self):
        while True:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.output_dir / "bjobs" / f"run_{timestamp}"

            if not run_dir.exists():
                return run_dir

            time.sleep(0.1)

    def _save_processor(self, run_dir):
        path = run_dir / "processor.pkl"

        with open(path, "wb") as f:
            pickle.dump(self, f)

        return path

    def _num_recon_chunks(self):
        return math.prod(self.recon_grid_shape)

    def _recon_label(self, start, stop, n_chunks, chunks_per_job):
        width = len(str(n_chunks - 1))

        if chunks_per_job == 1:
            return f"{start:0{width}d}"

        return f"{start:0{width}d}_{stop:0{width}d}"

    def _minutes_to_walltime(self, minutes):
        minutes = math.ceil(minutes)
        hours = minutes // 60
        remaining_minutes = minutes % 60

        return f"{hours}:{remaining_minutes:02d}"

    def _scaled_write_walltime(self, safety_factor=2.0, minimum_minutes=10):
        benchmark_size = math.prod(self.BENCHMARK_PROJ_SHAPE)
        current_size = math.prod(self.proj_shape)

        minutes = self.BENCHMARK_WRITE_MINUTES * current_size / benchmark_size
        minutes = max(minimum_minutes, minutes * safety_factor)

        return self._minutes_to_walltime(minutes)

    def _pad_factor_walltime_multiplier(self):
        return 1 + self.PAD_FACTOR_WALLTIME_SLOPE * (self.pad_factor or 0)

    def _pad_factor_memory_multiplier(self):
        return 1 + self.PAD_FACTOR_MEMORY_SLOPE * (self.pad_factor or 0)

    def _scaled_recon_memory(self):
        memory_gb = self.BENCHMARK_RECON_MEMORY_GB * self._pad_factor_memory_multiplier()
        return f"{math.ceil(memory_gb)}GB"

    def _scaled_recon_walltime(self, chunks_per_job=1, safety_factor=2.0, minimum_minutes=10):
        frac1 = math.prod(self.recon_chunk_shape) / math.prod(self.BENCHMARK_RECON_CHUNK_SHAPE)
        frac2 = math.prod(self.proj_shape) / math.prod(self.BENCHMARK_PROJ_SHAPE)
        frac = frac1 * frac2

        minutes_per_chunk = (
            self.BENCHMARK_RECON_MINUTES * frac * self._pad_factor_walltime_multiplier()
        )
        minutes = chunks_per_job * minutes_per_chunk
        minutes = max(minimum_minutes, minutes * safety_factor)

        return self._minutes_to_walltime(minutes)

    def _status_rows(self):
        result = subprocess.run(
            shlex.split(self.STATUS_COMMAND),
            check=False,
            capture_output=True,
            text=True,
        )

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

        if not lines:
            return [], []

        header = [value.strip() for value in lines[0].split(self.STATUS_DELIMITER)]
        rows = []

        for line in lines[1:]:
            values = [value.strip() for value in line.split(self.STATUS_DELIMITER)]

            if len(values) == len(header):
                rows.append(dict(zip(header, values, strict=True)))

        return header, rows

    def _matching_jobs(self, job_name):
        header, rows = self._status_rows()

        if "JOB_NAME" in header:
            matching = [row for row in rows if job_name in row["JOB_NAME"]]
        else:
            matching = [row for row in rows if job_name in " ".join(row.values())]

        return header, matching

    def _unfinished_jobs(self, job_name):
        header, rows = self._matching_jobs(job_name)
        unfinished = [row for row in rows if row.get("STAT") in self.ACTIVE_STATUSES]

        return header, unfinished

    def _format_rows(self, header, rows):
        if not rows:
            return ""

        widths = {col: max(len(col), max(len(row[col]) for row in rows)) for col in header}

        lines = [
            "  ".join(col.ljust(widths[col]) for col in header),
            "  ".join("-" * widths[col] for col in header),
        ]

        for row in rows:
            lines.append("  ".join(row[col].ljust(widths[col]) for col in header))

        return "\n".join(lines)

    def check_jobs(self, job_name, active_only=True, ret=False, pri=True):
        """
        Check matching batch jobs.

        By default, only unfinished jobs are shown. Use active_only=False to
        include recently finished DONE/EXIT jobs.
        """
        if active_only:
            header, rows = self._unfinished_jobs(job_name)
            message = f"No unfinished jobs found matching {job_name!r}."
        else:
            header, rows = self._matching_jobs(job_name)
            message = f"No active or recently finished jobs found matching {job_name!r}."

        if pri:
            if rows:
                print(self._format_rows(header, rows))
            else:
                print(message)

        if ret:
            return rows

    def clean_jobs(self, job_name, dry_run=False):
        """
        Kill unfinished batch jobs matching job_name.

        This is useful when the write job does not finish successfully and dependent reconstruction
        jobs remain pending.
        """
        header, unfinished = self._unfinished_jobs(job_name)

        if not unfinished:
            print(f"No unfinished jobs found matching {job_name!r}.")
            return

        print("Jobs selected for cleanup:")
        print(self._format_rows(header, unfinished))

        job_ids = [row["JOBID"] for row in unfinished]

        if dry_run:
            print("Dry run. Would run:")
            print("bkill " + " ".join(job_ids))
            return

        result = subprocess.run(
            ["bkill", *job_ids],
            check=False,
            capture_output=True,
            text=True,
        )

        if result.stdout.strip():
            print(result.stdout.strip())

        if result.stderr.strip():
            print(result.stderr.strip())

    @staticmethod
    def _parse_job_id(bsub_output):
        match = re.search(r"Job <(\d+)>", bsub_output)

        if match is None:
            raise RuntimeError(f"Could not parse job id from:\n{bsub_output}")

        return match.group(1)

    def _submit(self, script_path, dependency=None):
        cmd = ["bsub"]

        if dependency is not None:
            cmd += ["-w", f"done({dependency})"]

        with open(script_path, "rb") as f:
            result = subprocess.run(
                cmd,
                stdin=f,
                check=True,
                capture_output=True,
                text=True,
            )

        return self._parse_job_id(result.stdout)

    def _inline_python(self, processor_path, call):
        return f"""{self.python_executable} -u <<'PY'
import cloudpickle as pickle

with open({str(processor_path)!r}, 'rb') as f:
    processor = pickle.load(f)

{call}
PY"""

    def _script(
        self,
        *,
        job_name,
        queue,
        cores,
        memory,
        walltime,
        out_path,
        err_path,
        command,
        gpu,
    ):
        gpu_select_line = f'#BSUB -R "select[{self.RECON_GPU_SELECT}]"' if gpu else ""
        gpu_line = '#BSUB -gpu "num=1:mode=exclusive_process"' if gpu else ""
        module_lines = "".join(f"module load {m}\n" for m in self.GPU_MODULES)
        cuda_lines = f"{module_lines}nvidia-smi" if gpu else ""

        return f"""#!/bin/sh
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -n {cores}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem={memory}]"
#BSUB -W {walltime}
{gpu_select_line}
{gpu_line}
#BSUB -o {out_path}
#BSUB -e {err_path}

echo "Host: $HOSTNAME"

{cuda_lines}

{command}
"""

    def submit_jobs(
        self,
        job_name,
        *,
        write_projections=True,
        reconstruct=True,
        chunks_per_job=1,
        write_cores=16,
        recon_cores=8,
        write_memory="8GB",
        recon_memory=None,
        write_walltime=None,
        recon_walltime=None,
        walltime_safety_factor=1.5,
        check_existing=True,
        force=False,
        dry_run=False,
    ):
        """
        Submit DTU batch jobs.

        job_name is used in batch job names and in info.txt.
        Batch artifacts are written to output_dir/bjobs/run_<timestamp>.

        If write_walltime or recon_walltime is None, walltime is estimated from:
            8 angle chunks
            projections: 4713 x 3000 x 3000 took 15 min
            recon chunk: 1500^3 took 15 min

        If recon_memory is None, it is estimated as BENCHMARK_RECON_MEMORY_GB.

        recon_walltime and recon_memory are both further scaled by pad_factor (padding widens every
        projection before FDK) -- see PAD_FACTOR_WALLTIME_SLOPE / PAD_FACTOR_MEMORY_SLOPE.

        The estimated walltimes are multiplied by walltime_safety_factor. Memory estimates are not
        further scaled by walltime_safety_factor.

        The walltimes and memory can also be set manually, in which case no heuristic is applied.
        """
        if chunks_per_job < 1:
            raise ValueError("chunks_per_job must be at least 1.")

        if not write_projections and not reconstruct:
            raise ValueError("Nothing to submit.")

        if write_walltime is None:
            write_walltime = self._scaled_write_walltime(
                safety_factor=walltime_safety_factor,
            )

        if recon_walltime is None:
            recon_walltime = self._scaled_recon_walltime(
                chunks_per_job=chunks_per_job,
                safety_factor=walltime_safety_factor,
            )

        if recon_memory is None:
            recon_memory = self._scaled_recon_memory()

        if check_existing:
            header, unfinished = self._unfinished_jobs(job_name)

            if unfinished and not force:
                raise RuntimeError(
                    f"Found unfinished jobs matching {job_name!r}:\n"
                    + self._format_rows(header, unfinished)
                    + "\n\nUse force=True to submit anyway."
                )

        run_dir = self._new_run_dir()
        script_dir = run_dir / "scripts"
        log_dir = run_dir / "logs"

        script_dir.mkdir(parents=True)
        log_dir.mkdir()

        processor_path = self._save_processor(run_dir)

        write_job_id = None
        recon_job_ids = []

        if write_projections:
            job = f"{job_name}_write"
            script_path = script_dir / "write_projections.sh"

            command = self._inline_python(
                processor_path,
                "processor.write_projection_chunks()",
            )

            script = self._script(
                job_name=job,
                queue=self.WRITE_QUEUE,
                cores=write_cores,
                memory=write_memory,
                walltime=write_walltime,
                out_path=log_dir / "write_projections.out",
                err_path=log_dir / "write_projections.err",
                command=command,
                gpu=False,
            )

            script_path.write_text(script)

            if not dry_run:
                write_job_id = self._submit(script_path)

        if reconstruct:
            n_chunks = self._num_recon_chunks()

            for start in range(0, n_chunks, chunks_per_job):
                stop = min(start + chunks_per_job, n_chunks)
                label = self._recon_label(start, stop, n_chunks, chunks_per_job)

                job = f"{job_name}_recon_{label}"
                script_path = script_dir / f"recon_{label}.sh"

                command = self._inline_python(
                    processor_path,
                    f"processor.reconstruct(chunks=slice({start}, {stop}), overwrite_on_mismatch=True)",
                )

                script = self._script(
                    job_name=job,
                    queue=self.RECON_QUEUE,
                    cores=recon_cores,
                    memory=recon_memory,
                    walltime=recon_walltime,
                    out_path=log_dir / f"recon_{label}.out",
                    err_path=log_dir / f"recon_{label}.err",
                    command=command,
                    gpu=True,
                )

                script_path.write_text(script)

                if not dry_run:
                    if write_job_id is None:
                        existing_write_jobs = (
                            self.check_jobs(f"{job_name}_write", ret=True, pri=False) or []
                        )
                        if existing_write_jobs:
                            write_job_id = existing_write_jobs[0]["JOBID"]

                    recon_job_ids.append(
                        self._submit(
                            script_path,
                            dependency=write_job_id,
                        )
                    )

        info_path = run_dir / "info.txt"
        info_path.write_text(
            "\n".join(
                [
                    f"job_name: {job_name}",
                    f"dry_run: {dry_run}",
                    f"output_dir: {self.output_dir}",
                    f"processor_path: {processor_path}",
                    f"proj_shape: {self.proj_shape}",
                    f"recon_grid_shape: {self.recon_grid_shape}",
                    f"recon_chunk_shape: {self.recon_chunk_shape}",
                    f"vol_chunk_shape: {self.vol_chunk_shape}",
                    f"chunks_per_job: {chunks_per_job}",
                    f"write_queue: {self.WRITE_QUEUE}",
                    f"recon_queue: {self.RECON_QUEUE}",
                    f"recon_gpu_select: {self.RECON_GPU_SELECT}",
                    f"write_memory: {write_memory}",
                    f"recon_memory: {recon_memory}",
                    f"write_walltime: {write_walltime}",
                    f"recon_walltime: {recon_walltime}",
                    f"walltime_safety_factor: {walltime_safety_factor}",
                    f"write_job_id: {write_job_id}",
                    f"recon_job_ids: {recon_job_ids}",
                ]
            )
            + "\n"
        )

        if dry_run:
            print(f"Prepared batch job scripts for {job_name!r}.")
        else:
            print(f"Submitted batch jobs for {job_name!r}.")

        print(f"Run directory: {run_dir}")
        print(f"Batch job logs: {log_dir}")
        print(f"Info about run: {info_path}")

        if not dry_run:
            print(f"Check jobs with: processor.check_jobs({job_name!r})")

        print("If a job times out, inspect the matching .out/.err file in the logs folder.")
        print("If necessary, rerun with manual walltime values or a larger walltime_safety_factor.")

        return run_dir

    ###
    @staticmethod
    def _read_info(run_dir):
        run_dir = Path(run_dir)
        info_path = run_dir / "info.txt"

        info = {}

        with open(info_path) as f:
            for line in f:
                line = line.strip()

                if not line or ": " not in line:
                    continue

                key, value = line.split(": ", 1)
                info[key] = value

        if info.get("write_job_id") not in {None, "None"}:
            info["write_job_id"] = info["write_job_id"]
        else:
            info["write_job_id"] = None

        info["recon_job_ids"] = ast.literal_eval(info.get("recon_job_ids", "[]"))

        return info

    def _status_rows_for_job_ids(self, job_ids):
        job_ids = [str(job_id) for job_id in job_ids if job_id is not None]

        if not job_ids:
            return [], []

        cmd = shlex.split(self.STATUS_COMMAND) + job_ids

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

        if not lines:
            return [], []

        header = [value.strip() for value in lines[0].split(self.STATUS_DELIMITER)]
        rows = []

        for line in lines[1:]:
            values = [value.strip() for value in line.split(self.STATUS_DELIMITER)]

            if len(values) == len(header):
                rows.append(dict(zip(header, values, strict=True)))

        return header, rows

    def _run_jobs(self, run_dir):
        info = self._read_info(run_dir)
        job_ids = []

        if info["write_job_id"] is not None:
            job_ids.append(info["write_job_id"])

        job_ids.extend(info["recon_job_ids"])

        return self._status_rows_for_job_ids(job_ids)

    def summarize_run(self, run_dir, show_table=True):
        """
        Summarize one submitted run from its run directory.
        """
        info = self._read_info(run_dir)
        header, rows = self._run_jobs(run_dir)

        if not rows:
            print(f"No jobs found for run directory: {run_dir}")
            return

        write_job_id = info["write_job_id"]
        recon_job_ids = set(str(job_id) for job_id in info["recon_job_ids"])

        write_rows = [row for row in rows if row.get("JOBID") == str(write_job_id)]

        recon_rows = [row for row in rows if row.get("JOBID") in recon_job_ids]

        proj_zarr_chunk_dir = self.proj_zarr_path / "c"
        chunks_written = (
            sum(1 for p in proj_zarr_chunk_dir.iterdir() if p.is_dir())
            if proj_zarr_chunk_dir.exists()
            else 0
        )
        if chunks_written == self.angle_chunks:
            write_status = "DONE"
        else:
            write_status = write_rows[0].get("STAT") if write_rows else "UNKNOWN"

        recon_done = sum(row.get("STAT") == "DONE" for row in recon_rows)
        recon_exit = sum(row.get("STAT") == "EXIT" for row in recon_rows)
        recon_run = sum(row.get("STAT") == "RUN" for row in recon_rows)
        recon_pend = sum(row.get("STAT") == "PEND" for row in recon_rows)
        recon_unfinished = sum(row.get("STAT") in self.ACTIVE_STATUSES for row in recon_rows)

        print(f"Run directory: {run_dir}")
        print(f"Job name: {info.get('job_name', '-')}")
        print(
            f"Write projections: {write_status} ({chunks_written}/{self.angle_chunks} chunks written)"
        )
        print(f"Reconstruction chunks done: {recon_done}/{len(recon_job_ids)}")

        if recon_exit:
            print(f"Reconstruction chunks failed: {recon_exit}/{len(recon_job_ids)}")

        if recon_run or recon_pend:
            print(f"Reconstruction chunks running: {recon_run}")
            print(f"Reconstruction chunks pending: {recon_pend}")

        if recon_unfinished == 0 and recon_exit == 0 and recon_done == len(recon_job_ids):
            print("Summary: run completed successfully.")
        elif write_status == "EXIT":
            print("Summary: write_projections failed; reconstruction will not complete.")
        elif recon_exit:
            print("Summary: at least one reconstruction chunk failed.")
        elif recon_unfinished:
            print("Summary: run is still active.")
        else:
            print("Summary: run has no unfinished jobs, but did not complete cleanly.")

        if show_table:
            print()
            print(self._format_rows(header, rows))
