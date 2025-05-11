"""
Simplified API for Docker operations.
"""

import time
from typing import Dict, List, Optional, Union

from docker.models.containers import Container

from python_docker_app.docker_manager import DockerManager
from python_docker_app.running_container import RunningContainer
from python_docker_app.volume import Volume


class DockerApp:
    """
    A simplified interface for Docker operations.

    This class provides a high-level API for common Docker operations
    like pulling images and running containers, abstracting away the
    complexity of the underlying Docker SDK.
    """

    def __init__(
        self, image_name: str, container_name: str, tag: str = "latest"
    ) -> None:
        """
        Initialize a DockerApp instance.

        Args:
            image_name: The name of the Docker image
            container_name: The name to give to the container
            tag: The image tag to use (default: "latest")
        """
        self.image_name = image_name
        self.container_name = container_name
        self.tag = tag
        self.docker_manager = DockerManager()
        self.running_container: Optional[RunningContainer] = None
        self.container: Optional[Container] = None

    def pull(self, upgrade: bool = False) -> bool:
        """
        Pull the Docker image.

        Args:
            upgrade: If True, pull even if the image exists locally

        Returns:
            True if a new image was pulled, False if using existing image
        """
        return self.docker_manager.validate_or_download_image(
            self.image_name, self.tag, upgrade=upgrade
        )

    def run(
        self,
        command: Optional[str] = None,
        volumes: Optional[List[Volume]] = None,
        ports: Optional[Dict[int, int]] = None,
        detached: bool = True,
        remove_previous: bool = False,
    ) -> Union[Container, None]:
        """
        Run the Docker container.

        Args:
            command: The command to run in the container
            volumes: List of Volume objects for container volume mappings
            ports: Dict mapping host ports to container ports
            detached: If True, run in detached mode; if False, run interactively
            remove_previous: If True, remove any existing container with the same name

        Returns:
            The Container object if running in detached mode, None otherwise
        """
        if detached:
            self.container = self.docker_manager.run_container_detached(
                self.image_name,
                self.tag,
                self.container_name,
                command=command,
                volumes=volumes,
                ports=ports,
                remove_previous=remove_previous,
            )
            return self.container
        else:
            self.docker_manager.run_container_interactive(
                self.image_name,
                self.tag,
                self.container_name,
                command=command,
                volumes=volumes,
                ports=ports,
            )
            return None

    def attach(self) -> RunningContainer:
        """
        Attach to the running container and monitor its logs.

        Returns:
            A RunningContainer object that can be used to monitor and control the container
        """
        if self.container is None:
            self.container = self.docker_manager.get_container(self.container_name)
            if self.container is None:
                raise ValueError(f"Container {self.container_name} not found")

        self.running_container = self.docker_manager.attach_and_run(self.container)
        return self.running_container

    def stop(self) -> None:
        """
        Stop the running container.
        """
        if self.running_container:
            self.running_container.stop()
            self.running_container = None

        container = self.docker_manager.get_container(self.container_name)
        if container:
            self.docker_manager.suspend_container(container)

    def is_running(self) -> bool:
        """
        Check if the container is running.

        Returns:
            True if the container is running, False otherwise
        """
        return self.docker_manager.is_container_running(self.container_name)


def example_usage():
    """Example of how to use the DockerApp class."""
    try:
        # Create a DockerApp instance for a Python web server
        app = DockerApp(
            image_name="python", container_name="python-web-server", tag="3.10-slim"
        )

        # Pull the image (upgrade if needed)
        app.pull(upgrade=True)

        # Run the container with a simple HTTP server
        app.run(
            command="python -m http.server 8080",
            ports={8080: 8080},  # Map host port 8080 to container port 8080
            detached=True,
        )

        # Attach to the container to monitor logs
        running_container = app.attach()
        assert running_container is not None, "Failed to attach to the container"

        print("Container is running. Press Ctrl+C to stop.")

        # Wait for keyboard interrupt
        try:
            while app.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping container...")

        # Stop the container
        app.stop()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    example_usage()
