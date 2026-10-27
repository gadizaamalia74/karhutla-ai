import os, sys, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "model", "model_rf.joblib")

if not os.path.exists(MODEL):
    src = os.path.join(ROOT, "src", "pipeline.py")
    env = os.environ.copy()
    env["KARHUTLA_RAW"] = os.path.join(ROOT, "data", "raw")
    result = subprocess.run(
        [sys.executable, src],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("=== STDOUT pipeline ===")
        print(result.stdout[-3000:])
        print("=== STDERR pipeline ===")
        print(result.stderr[-3000:])
        raise RuntimeError("Pipeline gagal. Lihat log di atas.")
    else:
        print("Pipeline selesai.")
        print(result.stdout[-1000:])