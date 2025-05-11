from dataclasses import dataclass


@dataclass
class Volume:
    """
    Represents a Docker volume mapping between host and container.

    Attributes:
        host_path: Path on the host system (e.g., "C:\\Users\\username\\project")
        container_path: Path inside the container (e.g., "/app/data")
        mode: Access mode, "rw" for read-write or "ro" for read-only
    """

    host_path: str
    container_path: str
    mode: str = "rw"

    def to_dict(self) -> dict[str, dict[str, str]]:
        """Convert the Volume object to the format expected by Docker API."""
        return {self.host_path: {"bind": self.container_path, "mode": self.mode}}

    @classmethod
    def from_dict(cls, volume_dict: dict[str, dict[str, str]]) -> list["Volume"]:
        """Create Volume objects from a Docker volume dictionary."""
        volumes = []
        for host_path, config in volume_dict.items():
            volumes.append(
                cls(
                    host_path=host_path,
                    container_path=config["bind"],
                    mode=config.get("mode", "rw"),
                )
            )
        return volumes
