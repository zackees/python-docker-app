import sys
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


def hack_to_fix_mac(volumes: list[Volume] | None) -> list[Volume] | None:
    """Fixes the volume mounts on MacOS by removing the mode."""
    if volumes is None:
        return None
    if sys.platform != "darwin":
        # Only macos needs hacking.
        return volumes

    volumes = volumes.copy()
    # Work around a Docker bug on MacOS where the expected network socket to the
    # the host is not mounted correctly. This was actually fixed in recent versions
    # of docker client but there is a large chunk of Docker clients out there with
    # this bug in it.
    #
    # This hack is done by mounting the socket directly to the container.
    # This socket talks to the docker daemon on the host.
    #
    # Found here.
    # https://github.com/docker/docker-py/issues/3069#issuecomment-1316778735
    # if it exists already then return the input
    for volume in volumes:
        if volume.host_path == "/var/run/docker.sock":
            return volumes
    # ok it doesn't exist, so add it
    volumes.append(
        Volume(
            host_path="/var/run/docker.sock",
            container_path="/var/run/docker.sock",
            mode="rw",
        )
    )
    return volumes
