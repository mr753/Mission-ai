from typing import Protocol, List
from pathlib import Path

class ImageSourceResolver(Protocol):
    def resolve(self, url: str) -> List[Path]:
        ...
