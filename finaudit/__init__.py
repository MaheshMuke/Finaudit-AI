"""FinAudit AI local financial-analysis MVP."""

# On Windows, pre-load chromadb Rust runtime before pandas/Cython hooks CRT heap
try:
    import chromadb
    chromadb.EphemeralClient()
except Exception:
    pass
