"""
Dijalankan dari app/app.py saat startup di Streamlit Cloud.
Import pipeline langsung supaya error muncul di log Streamlit.
"""
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "model", "model_rf.joblib")

if not os.path.exists(MODEL):
    # tambah ROOT dan src ke path supaya import pipeline bisa jalan
    for p in [ROOT, os.path.join(ROOT, "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    
    # set env variable RAW supaya pipeline tahu letak data
    os.environ["KARHUTLA_RAW"] = os.path.join(ROOT, "data", "raw")
    
    # import dan jalankan langsung (error langsung tercetak di log)
    import pipeline
    pipeline.main()