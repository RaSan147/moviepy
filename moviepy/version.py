try:
    from importlib.metadata import version

    __version__ = version("moviepy")
except Exception:
    __version__ = "2.1.2.dev3"  # Fallback version if import fails
