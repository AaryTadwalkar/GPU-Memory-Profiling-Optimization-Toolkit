"""
File: run.py

1. What this file does:
   The unified command-line entry point for the GPU Memory Profiling & Optimization Toolkit.
   Provides convenient commands to launch the Streamlit dashboard, run automated tests,
   generate synthetic mock datasets, validate/normalize telemetry pipelines, and execute
   profiling analysis engines.

2. Overall design:
   Uses `argparse` with subcommands (`dashboard`, `test`, `mock`, `pipeline`, `pressure`,
   `fragmentation`, `recommendations`, `contention`). Ensures that developers and
   evaluators can drive the entire toolkit from one root script without remembering
   deep submodule paths.

3. Elements in the file:
   - `cmd_dashboard`: Launches Streamlit web dashboard.
   - `cmd_test`: Executes automated test suite.
   - `cmd_mock`: Generates mock datasets.
   - `cmd_pipeline`: Audits, normalizes, and summarizes a telemetry CSV.
   - `cmd_pressure`: Runs memory pressure analysis on a CSV.
   - `cmd_fragmentation`: Runs FSFI analysis on snapshot files.
   - `cmd_recommendations`: Runs recommendation engine on a CSV.
   - `cmd_contention`: Runs multi-process contention analysis on a CSV.
   - Main CLI parser.

4. Final output of the file:
   Launches requested toolkit service or prints structured analysis to console.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_python_exe() -> str:
    """Return the path to the virtual environment Python if available."""
    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.is_file():
        return str(venv_py)
    venv_py_nix = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_py_nix.is_file():
        return str(venv_py_nix)
    return sys.executable


def get_streamlit_exe() -> str:
    """Return the path to the Streamlit executable."""
    venv_st = PROJECT_ROOT / ".venv" / "Scripts" / "streamlit.exe"
    if venv_st.is_file():
        return str(venv_st)
    venv_st_nix = PROJECT_ROOT / ".venv" / "bin" / "streamlit"
    if venv_st_nix.is_file():
        return str(venv_st_nix)
    return "streamlit"


def cmd_dashboard(args):
    """Launch the Streamlit web dashboard."""
    app_path = PROJECT_ROOT / "dashboard" / "app.py"
    py_exe = get_python_exe()
    port = args.port or 8501
    cmd = [py_exe, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    print(f"[Toolkit] Launching Streamlit dashboard on port {port}...")
    print(f"[Toolkit] Command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def cmd_test(args):
    """Run automated unit tests."""
    from run_tests import run_all_tests
    sys.exit(run_all_tests())


def cmd_mock(args):
    """Generate synthetic mock datasets for CPU testing and dashboard demos."""
    from data.mock_generator import generate_all_mock_data
    out_dir = args.out_dir or "data/mock"
    generate_all_mock_data(out_dir)


def cmd_pipeline(args):
    """Validate, normalize, and summarize a telemetry CSV."""
    from pipeline.normalize import load_and_normalize_csv
    from pipeline.validate import validate_csv_file
    from pipeline.aggregate import compute_profile_summary

    csv_path = Path(args.csv_path)
    print(f"[Pipeline] Processing: {csv_path}")

    # Step 1: Validation
    _, val_res = validate_csv_file(csv_path)
    print(f"[Pipeline] {val_res.summary()}")
    if val_res.errors:
        print(f"[Pipeline] First 3 errors: {val_res.errors[:3]}")

    # Step 2: Normalization
    df = load_and_normalize_csv(csv_path)
    print(f"[Pipeline] Normalized {len(df)} rows into DataFrame.")

    # Step 3: Aggregation
    summary = compute_profile_summary(df, budget_bytes=args.budget)
    print("[Pipeline] Profile Summary:")
    print(f"  • Total Events Analyzed : {summary.total_events}")
    print(f"  • Duration              : {summary.duration_s} seconds")
    print(f"  • Sampling Interval     : {summary.sampling_interval_s} seconds")
    if summary.peak_allocated_bytes:
        print(f"  • Peak Allocated Memory : {summary.peak_allocated_bytes / (1024*1024):.2f} MiB")
    if summary.peak_reserved_bytes:
        print(f"  • Peak Reserved Memory  : {summary.peak_reserved_bytes / (1024*1024):.2f} MiB")
    if summary.peak_device_used_bytes:
        print(f"  • Peak Device Used VRAM : {summary.peak_device_used_bytes / (1024*1024):.2f} MiB")
    if summary.pids_observed:
        print(f"  • PIDs Observed         : {summary.pids_observed}")


def cmd_pressure(args):
    """Run memory pressure and heuristic OOM-risk analysis."""
    py_exe = get_python_exe()
    cmd = [py_exe, "profiler/memory_pressure.py", args.csv_path, "--budget", str(args.budget)]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def cmd_fragmentation(args):
    """Run FSFI fragmentation analysis."""
    py_exe = get_python_exe()
    cmd = [py_exe, "profiler/fragmentation.py", args.csv_path]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def cmd_recommendations(args):
    """Run traceable optimization recommendation engine."""
    py_exe = get_python_exe()
    cmd = [py_exe, "optimizer/recommendations.py", args.csv_path, "--budget", str(args.budget)]
    if args.out:
        cmd.extend(["--out", args.out])
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def cmd_contention(args):
    """Run multi-process contention analysis."""
    py_exe = get_python_exe()
    cmd = [py_exe, "profiler/contention.py", args.csv_path]
    if args.out:
        cmd.extend(["--out", args.out])
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="GPU Memory Profiling & Optimization Toolkit - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch Streamlit web dashboard")
    p_dash.add_argument("--port", type=int, default=8501, help="Port to bind dashboard (default: 8501)")
    p_dash.set_defaults(func=cmd_dashboard)

    # test
    p_test = subparsers.add_parser("test", help="Run automated unit test suite")
    p_test.set_defaults(func=cmd_test)

    # mock
    p_mock = subparsers.add_parser("mock", help="Generate synthetic mock profiling datasets")
    p_mock.add_argument("--out-dir", default="data/mock", help="Output directory (default: data/mock)")
    p_mock.set_defaults(func=cmd_mock)

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Audit, normalize, and summarize a telemetry CSV")
    p_pipe.add_argument("csv_path", help="Path to MemoryEvent CSV file")
    p_pipe.add_argument("--budget", type=int, default=6442450944, help="GPU budget in bytes (default: 6 GiB)")
    p_pipe.set_defaults(func=cmd_pipeline)

    # pressure
    p_press = subparsers.add_parser("pressure", help="Analyze memory pressure and heuristic OOM risk")
    p_press.add_argument("csv_path", help="Path to torch_events.csv")
    p_press.add_argument("--budget", type=int, default=6442450944, help="Workload budget in bytes (default: 6 GiB)")
    p_press.set_defaults(func=cmd_pressure)

    # fragmentation
    p_frag = subparsers.add_parser("fragmentation", help="Analyze allocator-level FSFI fragmentation")
    p_frag.add_argument("csv_path", help="Path to torch_events.csv containing snapshot events")
    p_frag.set_defaults(func=cmd_fragmentation)

    # recommendations
    p_recs = subparsers.add_parser("recommendations", help="Generate traceable optimization recommendations")
    p_recs.add_argument("csv_path", help="Path to MemoryEvent CSV file")
    p_recs.add_argument("--budget", type=int, default=6442450944, help="GPU budget in bytes (default: 6 GiB)")
    p_recs.add_argument("--out", help="Optional path to output JSON report")
    p_recs.set_defaults(func=cmd_recommendations)

    # contention
    p_cont = subparsers.add_parser("contention", help="Analyze multi-process GPU resource contention")
    p_cont.add_argument("csv_path", help="Path to contention MemoryEvent CSV")
    p_cont.add_argument("--out", help="Optional path to output JSON report")
    p_cont.set_defaults(func=cmd_contention)

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
