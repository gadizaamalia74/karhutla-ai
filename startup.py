import os, sys, subprocess

MODEL = os.path.join(os.path.dirname(__file__), "model", "model_rf.joblib")
if not os.path.exists(MODEL):
    src = os.path.join(os.path.dirname(__file__), "src", "pipeline.py")
    subprocess.run([sys.executable, src], check=True)