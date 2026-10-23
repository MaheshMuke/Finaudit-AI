"""Pytest root configuration and Windows CRT compatibility initialization."""

# Pre-initialize ChromaDB Rust runtime before pandas/Cython alters MSVC CRT heap on Windows
try:
    import chromadb
    chromadb.EphemeralClient()
except Exception:
    pass
