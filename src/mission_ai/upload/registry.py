from typing import Dict, Type, List
from mission_ai.upload.base import BasePlatformAdapter

_ADAPTER_REGISTRY: Dict[str, Type[BasePlatformAdapter]] = {}

def register_adapter(platform: str, adapter_cls: Type[BasePlatformAdapter]) -> None:
    """Register a platform adapter class."""
    _ADAPTER_REGISTRY[platform.lower()] = adapter_cls

def get_platform_adapter(platform: str) -> BasePlatformAdapter:
    """Retrieve an instance of the adapter for the given platform."""
    key = platform.lower()
    adapter_cls = _ADAPTER_REGISTRY.get(key)
    if not adapter_cls:
        raise ValueError(f"Unsupported or unregistered platform adapter: {platform}")
    return adapter_cls()

def list_supported_platforms() -> List[str]:
    """Return a list of all registered supported platforms (unique names)."""
    # Return unique base platform names or registered keys
    seen = set()
    platforms = []
    for k, cls in _ADAPTER_REGISTRY.items():
        inst = cls()
        if inst.platform_name not in seen:
            seen.add(inst.platform_name)
            platforms.append(inst.platform_name)
    return sorted(platforms)
