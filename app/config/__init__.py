# app/config/__init__.py
# Re-exports the public surface of the config package.
from app.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
