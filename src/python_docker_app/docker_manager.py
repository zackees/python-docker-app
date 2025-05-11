"""
New abstraction for Docker management with improved Ctrl+C handling.
"""

import _thread
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import docker
from appdirs import user_data_dir
from disklru import DiskLRUCache
from docker.client import DockerClient
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.models.images import Image
from filelock import FileLock

from python_docker_app.spinner import Spinner

CONFIG_DIR = Path(user_data_dir("python-docker-app", "python-docker-app"))
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = CONFIG_DIR / "db.db"
DISK_CACHE = DiskLRUCache(str(DB_FILE), 10)
_IS_GITHUB = "GITHUB_ACTIONS" in os.environ


# Docker uses datetimes in UTC but without the timezone info. If we pass in a tz
# then it will throw an exception.
def _utc_now_no_tz() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(tzinfo=None)


def _win32_docker_location() -> str | None:
    home_dir = Path.home()
    out = [
        "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe",
        f"{home_dir}\\AppData\\Local\\Docker\\Docker Desktop.exe",
    ]
    for loc in out:
        if Path(loc).exists():
            return loc
    return None


def get_lock(image_name: str) -> FileLock:
    """Get the file lock for this DockerManager instance."""
    lock_file = CONFIG_DIR / f"{image_name}.lock"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    print(CONFIG_DIR)
    if not lock_file.parent.exists():
        lock_file.parent.mkdir(parents=True, exist_ok=True)
    out: FileLock
    out = FileLock(str(lock_file))  # type: ignore
    return out


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


class RunningContainer:
    def __init__(
        self,
        container: Container,
        first_run: bool = False,
    ) -> None:
        self.container = container
        self.first_run = first_run
        self.running = True
        self.thread = threading.Thread(target=self._log_monitor)
        self.thread.daemon = True
        self.thread.start()

    def _log_monitor(self):
        from_date = _utc_now_no_tz() if not self.first_run else None
        to_date = _utc_now_no_tz()

        while self.running:
            try:
                for log in self.container.logs(
                    follow=False, since=from_date, until=to_date, stream=True
                ):
                    print(log.decode("utf-8"), end="")
                    # self.filter.print(log)
                time.sleep(0.1)
                from_date = to_date
                to_date = _utc_now_no_tz()
            except KeyboardInterrupt:
                print("Monitoring logs interrupted by user.")
                _thread.interrupt_main()
                break
            except Exception as e:
                print(f"Error monitoring logs: {e}")
                break

    def detach(self) -> None:
        """Stop monitoring the container logs"""
        self.running = False
        self.thread.join()

    def stop(self) -> None:
        """Stop the container"""
        self.container.stop()
        self.detach()


def _hack_to_fix_mac(volumes: list[Volume] | None) -> list[Volume] | None:
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


class DockerManager:
    def __init__(self) -> None:
        from docker.errors import DockerException

        self.is_suspended: bool = False

        try:
            self._client: DockerClient | None = None
            self.first_run = False
        except DockerException as e:
            stack = traceback.format_exc()
            warnings.warn(f"Error initializing Docker client: {e}\n{stack}")
            raise

    @property
    def client(self) -> DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    @staticmethod
    def is_docker_installed() -> bool:
        """Check if Docker is installed on the system."""
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            print("Docker is installed.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Docker command failed: {str(e)}")
            return False
        except FileNotFoundError:
            print("Docker is not installed.")
            return False

    @staticmethod
    def ensure_linux_containers_for_windows() -> bool:
        """Ensure Docker is using Linux containers on Windows."""
        if sys.platform != "win32":
            return True  # Only needed on Windows

        try:
            # Check if we're already in Linux container mode
            result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, check=True
            )

            if "linux" in result.stdout.lower():
                print("Already using Linux containers")
                return True

            if not _IS_GITHUB:
                answer = (
                    input(
                        "\nDocker on Windows must be in linux mode, this is a global change, switch? [y/n]"
                    )
                    .strip()
                    .lower()[:1]
                )
                if answer != "y":
                    return False

            print("Switching to Linux containers...")
            warnings.warn("Switching Docker to use Linux container context...")

            # Explicitly specify the Linux container context
            linux_context = "desktop-linux"
            subprocess.run(
                ["cmd", "/c", f"docker context use {linux_context}"],
                check=True,
                capture_output=True,
            )

            # Verify the switch worked
            verify = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, check=True
            )
            if "linux" in verify.stdout.lower():
                print(
                    f"Successfully switched to Linux containers using '{linux_context}' context"
                )
                return True

            warnings.warn(
                f"Failed to switch to Linux containers with context '{linux_context}': {verify.stdout}"
            )
            return False

        except subprocess.CalledProcessError as e:
            print(f"Failed to switch to Linux containers: {e}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error switching to Linux containers: {e}")
            return False

    @staticmethod
    def is_running() -> tuple[bool, Exception | None]:
        """Check if Docker is running by pinging the Docker daemon."""

        if not DockerManager.is_docker_installed():
            print("Docker is not installed.")
            return False, Exception("Docker is not installed.")
        try:
            # self.client.ping()
            client = docker.from_env()
            client.ping()
            print("Docker is running.")
            return True, None
        except DockerException as e:
            print(f"Docker is not running: {str(e)}")
            return False, e
        except Exception as e:
            print(f"Error pinging Docker daemon: {str(e)}")
            return False, e

    def start(self) -> bool:
        """Attempt to start Docker Desktop (or the Docker daemon) automatically."""
        print("Attempting to start Docker...")

        try:
            if sys.platform == "win32":
                docker_path = _win32_docker_location()
                if not docker_path:
                    print("Docker Desktop not found.")
                    return False
                subprocess.run(["start", "", docker_path], shell=True)
            elif sys.platform == "darwin":
                subprocess.run(["open", "-a", "Docker"])
            elif sys.platform.startswith("linux"):
                subprocess.run(["sudo", "systemctl", "start", "docker"])
            else:
                print("Unknown platform. Cannot auto-launch Docker.")
                return False

            # Wait for Docker to start up with increasing delays
            print("Waiting for Docker Desktop to start...")
            attempts = 0
            max_attempts = 20  # Increased max wait time
            while attempts < max_attempts:
                attempts += 1
                if self.is_running():
                    print("Docker started successfully.")
                    return True

                # Gradually increase wait time between checks
                wait_time = min(5, 1 + attempts * 0.5)
                print(
                    f"Docker not ready yet, waiting {wait_time:.1f}s... (attempt {attempts}/{max_attempts})"
                )
                time.sleep(wait_time)

            print("Failed to start Docker within the expected time.")
            print(
                "Please try starting Docker Desktop manually and run this command again."
            )
        except KeyboardInterrupt:
            print("Aborted by user.")
            raise
        except Exception as e:
            print(f"Error starting Docker: {str(e)}")
        return False

    def validate_or_download_image(
        self, image_name: str, tag: str = "latest", upgrade: bool = False
    ) -> bool:
        """
        Validate if the image exists, and if not, download it.
        If upgrade is True, will pull the latest version even if image exists locally.
        """
        ok = DockerManager.ensure_linux_containers_for_windows()
        if not ok:
            warnings.warn(
                "Failed to ensure Linux containers on Windows. This build may fail."
            )
        print(f"Validating image {image_name}:{tag}...")
        remote_image_hash_from_local_image: str | None = None
        remote_image_hash: str | None = None

        with get_lock(f"{image_name}-{tag}"):
            try:
                local_image = self.client.images.get(f"{image_name}:{tag}")
                print(f"Image {image_name}:{tag} is already available.")

                if upgrade:
                    remote_image = self.client.images.get_registry_data(
                        f"{image_name}:{tag}"
                    )
                    remote_image_hash = remote_image.id

                    try:
                        local_image_id = local_image.id
                        assert local_image_id is not None
                        remote_image_hash_from_local_image = DISK_CACHE.get(
                            local_image_id
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        remote_image_hash_from_local_image = None
                        stack = traceback.format_exc()
                        warnings.warn(
                            f"Error getting remote image hash from local image: {stack}"
                        )
                    if remote_image_hash_from_local_image == remote_image_hash:
                        print(f"Local image {image_name}:{tag} is up to date.")
                        return False

                    # Quick check for latest version
                    with Spinner(f"Pulling newer version of {image_name}:{tag}..."):
                        cmd_list = ["docker", "pull", f"{image_name}:{tag}"]
                        cmd_str = subprocess.list2cmdline(cmd_list)
                        print(f"Running command: {cmd_str}")
                        subprocess.run(cmd_list, check=True)
                    print(f"Updated to newer version of {image_name}:{tag}")
                    local_image_hash = self.client.images.get(f"{image_name}:{tag}").id
                    assert local_image_hash is not None
                    if remote_image_hash is not None:
                        DISK_CACHE.put(local_image_hash, remote_image_hash)
                    return True

            except ImageNotFound:
                print(f"Image {image_name}:{tag} not found.")
                with Spinner("Loading "):
                    # We use docker cli here because it shows the download.
                    cmd_list = ["docker", "pull", f"{image_name}:{tag}"]
                    cmd_str = subprocess.list2cmdline(cmd_list)
                    print(f"Running command: {cmd_str}")
                    subprocess.run(cmd_list, check=True)
                try:
                    local_image = self.client.images.get(f"{image_name}:{tag}")
                    local_image_hash = local_image.id
                    print(f"Image {image_name}:{tag} downloaded successfully.")
                except ImageNotFound:
                    warnings.warn(f"Image {image_name}:{tag} not found after download.")
        return True

    def tag_image(self, image_name: str, old_tag: str, new_tag: str) -> None:
        """
        Tag an image with a new tag.
        """
        image: Image = self.client.images.get(f"{image_name}:{old_tag}")
        image.tag(image_name, new_tag)
        print(f"Image {image_name}:{old_tag} tagged as {new_tag}.")

    def _container_configs_match(
        self,
        container: Container,
        command: str | None,
        volumes_dict: dict[str, dict[str, str]] | None,
        ports: dict[int, int] | None,
    ) -> bool:
        """Compare if existing container has matching configuration"""
        try:
            # Check if container is using the same image
            image = container.image
            assert image is not None
            container_image_id = image.id
            container_image_tags = image.tags
            assert container_image_id is not None

            # Simplified image comparison - just compare the IDs directly
            if not container_image_tags:
                print(f"Container using untagged image with ID: {container_image_id}")
            else:
                current_image = self.client.images.get(container_image_tags[0])
                if container_image_id != current_image.id:
                    print(
                        f"Container using different image version. Container: {container_image_id}, Current: {current_image.id}"
                    )
                    return False

            # Check command if specified
            if command and container.attrs["Config"]["Cmd"] != command.split():
                print(
                    f"Command mismatch: {container.attrs['Config']['Cmd']} != {command}"
                )
                return False

            # Check volumes if specified
            if volumes_dict:
                container_mounts = (
                    {
                        m["Source"]: {"bind": m["Destination"], "mode": m["Mode"]}
                        for m in container.attrs["Mounts"]
                    }
                    if container.attrs.get("Mounts")
                    else {}
                )

                for host_dir, mount in volumes_dict.items():
                    if host_dir not in container_mounts:
                        print(f"Volume {host_dir} not found in container mounts.")
                        return False
                    if container_mounts[host_dir] != mount:
                        print(
                            f"Volume {host_dir} has different mount options: {container_mounts[host_dir]} != {mount}"
                        )
                        return False

            # Check ports if specified
            if ports:
                container_ports = (
                    container.attrs["Config"]["ExposedPorts"]
                    if container.attrs["Config"].get("ExposedPorts")
                    else {}
                )
                container_port_bindings = (
                    container.attrs["HostConfig"]["PortBindings"]
                    if container.attrs["HostConfig"].get("PortBindings")
                    else {}
                )

                for container_port, host_port in ports.items():
                    port_key = f"{container_port}/tcp"
                    if port_key not in container_ports:
                        print(f"Container port {port_key} not found.")
                        return False
                    if not container_port_bindings.get(port_key, [{"HostPort": None}])[
                        0
                    ]["HostPort"] == str(host_port):
                        print(f"Port {host_port} is not bound to {port_key}.")
                        return False
        except KeyboardInterrupt:
            raise
        except NotFound:
            print("Container not found.")
            return False
        except Exception as e:
            stack = traceback.format_exc()
            warnings.warn(f"Error checking container config: {e}\n{stack}")
            return False
        return True

    def run_container_detached(
        self,
        image_name: str,
        tag: str,
        container_name: str,
        command: str | None = None,
        volumes: list[Volume] | None = None,
        ports: dict[int, int] | None = None,
        remove_previous: bool = False,
    ) -> Container:
        """
        Run a container from an image. If it already exists with matching config, start it.
        If it exists with different config, remove and recreate it.

        Args:
            volumes: List of Volume objects for container volume mappings
            ports: Dict mapping host ports to container ports
                    Example: {8080: 80} maps host port 8080 to container port 80
        """
        volumes = _hack_to_fix_mac(volumes)
        # Convert volumes to the format expected by Docker API
        volumes_dict = None
        if volumes is not None:
            volumes_dict = {}
            for volume in volumes:
                volumes_dict.update(volume.to_dict())

        # Serialize the volumes to a json string
        if volumes_dict:
            volumes_str = json.dumps(volumes_dict)
            print(f"Volumes: {volumes_str}")
            print("Done")
        image_name = f"{image_name}:{tag}"
        try:
            container: Container = self.client.containers.get(container_name)

            if remove_previous:
                print(f"Removing existing container {container_name}...")
                container.remove(force=True)
                raise NotFound("Container removed due to remove_previous")
            # Check if configuration matches
            elif not self._container_configs_match(
                container, command, volumes_dict, ports
            ):
                print(
                    f"Container {container_name} exists but with different configuration. Removing and recreating..."
                )
                container.remove(force=True)
                raise NotFound("Container removed due to config mismatch")
            print(f"Container {container_name} found with matching configuration.")

            # Existing container with matching config - handle various states
            if container.status == "running":
                print(f"Container {container_name} is already running.")
            elif container.status == "exited":
                print(f"Starting existing container {container_name}.")
                container.start()
            elif container.status == "restarting":
                print(f"Waiting for container {container_name} to restart...")
                timeout = 10
                container.wait(timeout=10)
                if container.status == "running":
                    print(f"Container {container_name} has restarted.")
                else:
                    print(
                        f"Container {container_name} did not restart within {timeout} seconds."
                    )
                    container.stop(timeout=0)
                    print(f"Container {container_name} has been stopped.")
                    container.start()
            elif container.status == "paused":
                print(f"Resuming existing container {container_name}.")
                container.unpause()
            else:
                print(f"Unknown container status: {container.status}")
                print(f"Starting existing container {container_name}.")
                self.first_run = True
                container.start()
        except NotFound:
            print(f"Creating and starting {container_name}")
            out_msg = f"# Running in container: {command}"
            msg_len = len(out_msg)
            print("\n" + "#" * msg_len)
            print(out_msg)
            print("#" * msg_len + "\n")
            container = self.client.containers.run(
                image=image_name,
                command=command,
                name=container_name,
                detach=True,
                tty=True,
                volumes=volumes_dict,
                ports=ports,  # type: ignore
                remove=True,
            )
        return container

    def run_container_interactive(
        self,
        image_name: str,
        tag: str,
        container_name: str,
        command: str | None = None,
        volumes: list[Volume] | None = None,
        ports: dict[int, int] | None = None,
    ) -> None:
        # Convert volumes to the format expected by Docker API
        volumes = _hack_to_fix_mac(volumes)
        volumes_dict = None
        if volumes is not None:
            volumes_dict = {}
            for volume in volumes:
                volumes_dict.update(volume.to_dict())
        # Remove existing container
        try:
            container: Container = self.client.containers.get(container_name)
            container.remove(force=True)
        except NotFound:
            pass
        start_time = time.time()
        try:
            docker_command: list[str] = [
                "docker",
                "run",
                "-it",
                "--rm",
                "--name",
                container_name,
            ]
            if volumes_dict:
                for host_dir, mount in volumes_dict.items():
                    docker_volume_arg = [
                        "-v",
                        f"{host_dir}:{mount['bind']}:{mount['mode']}",
                    ]
                    docker_command.extend(docker_volume_arg)
            if ports:
                for host_port, container_port in ports.items():
                    docker_command.extend(["-p", f"{host_port}:{container_port}"])
            docker_command.append(f"{image_name}:{tag}")
            if command:
                docker_command.append(command)
            cmd_str: str = subprocess.list2cmdline(docker_command)
            print(f"Running command: {cmd_str}")
            subprocess.run(docker_command, check=False)
        except subprocess.CalledProcessError as e:
            print(f"Error running Docker command: {e}")
            diff = time.time() - start_time
            if diff < 5:
                raise
            sys.exit(1)  # Probably a user exit.

    def attach_and_run(self, container: Container | str) -> RunningContainer:
        """
        Attach to a running container and monitor its logs in a background thread.
        Returns a RunningContainer object that can be used to stop monitoring.
        """
        if isinstance(container, str):
            container_name = container
            tmp = self.get_container(container)
            assert tmp is not None, f"Container {container_name} not found."
            container = tmp

        assert container is not None, "Container not found."

        print(f"Attaching to container {container.name}...")

        first_run = self.first_run
        self.first_run = False

        return RunningContainer(container, first_run)

    def suspend_container(self, container: Container | str) -> None:
        """
        Suspend (pause) the container.
        """
        if self.is_suspended:
            return
        if isinstance(container, str):
            container_name = container
            # container = self.get_container(container)
            tmp = self.get_container(container_name)
            if not tmp:
                print(f"Could not put container {container_name} to sleep.")
                return
            container = tmp
        assert isinstance(container, Container)
        try:
            if platform.system() == "Windows":
                container.pause()
            else:
                container.stop()
                container.remove()
            print(f"Container {container.name} has been suspended.")
        except KeyboardInterrupt:
            print(f"Container {container.name} interrupted by keyboard interrupt.")
        except Exception as e:
            print(f"Failed to suspend container {container.name}: {e}")

    def resume_container(self, container: Container | str) -> None:
        """
        Resume (unpause) the container.
        """
        container_name = "UNKNOWN"
        if isinstance(container, str):
            container_name = container
            container_or_none = self.get_container(container)
            if container_or_none is None:
                print(f"Could not resume container {container}.")
                return
            container = container_or_none
            container_name = container.name
        elif isinstance(container, Container):
            container_name = container.name
        assert isinstance(container, Container)
        if not container:
            print(f"Could not resume container {container}.")
            return
        try:
            assert isinstance(container, Container)
            container.unpause()
            print(f"Container {container.name} has been resumed.")
        except Exception as e:
            print(f"Failed to resume container {container_name}: {e}")

    def get_container(self, container_name: str) -> Container | None:
        """
        Get a container by name.
        """
        try:
            return self.client.containers.get(container_name)
        except NotFound:
            return None

    def is_container_running(self, container_name: str) -> bool:
        """
        Check if a container is running.
        """
        try:
            container = self.client.containers.get(container_name)
            return container.status == "running"
        except NotFound:
            print(f"Container {container_name} not found.")
            return False

    def build_image(
        self,
        image_name: str,
        tag: str,
        dockerfile_path: Path,
        build_context: Path,
        build_args: dict[str, str] | None = None,
        platform_tag: str = "",
    ) -> None:
        """
        Build a Docker image from a Dockerfile.

        Args:
            image_name: Name for the image
            tag: Tag for the image
            dockerfile_path: Path to the Dockerfile
            build_context: Path to the build context directory
            build_args: Optional dictionary of build arguments
            platform_tag: Optional platform tag (e.g. "-arm64")
        """
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")

        if not build_context.exists():
            raise FileNotFoundError(
                f"Build context directory not found at {build_context}"
            )

        print(f"Building Docker image {image_name}:{tag} from {dockerfile_path}")

        # Prepare build arguments
        buildargs = build_args or {}
        if platform_tag:
            buildargs["PLATFORM_TAG"] = platform_tag

        try:
            cmd_list = [
                "docker",
                "build",
                "-t",
                f"{image_name}:{tag}",
            ]

            # Add build args
            for arg_name, arg_value in buildargs.items():
                cmd_list.extend(["--build-arg", f"{arg_name}={arg_value}"])

            # Add dockerfile and context paths
            cmd_list.extend(["-f", str(dockerfile_path), str(build_context)])

            cmd_str = subprocess.list2cmdline(cmd_list)
            print(f"Running command: {cmd_str}")

            # Run the build command
            # cp = subprocess.run(cmd_list, check=True, capture_output=True)
            proc: subprocess.Popen = subprocess.Popen(
                cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            stdout = proc.stdout
            assert stdout is not None, "stdout is None"
            for line in iter(stdout.readline, b""):
                try:
                    line_str = line.decode("utf-8")
                    print(line_str, end="")
                except UnicodeDecodeError:
                    print("Error decoding line")
            rtn = proc.wait()
            if rtn != 0:
                warnings.warn(
                    f"Error building Docker image, is docker running? {rtn}, stdout: {stdout}, stderr: {proc.stderr}"
                )
                raise subprocess.CalledProcessError(rtn, cmd_str)
            print(f"Successfully built image {image_name}:{tag}")

        except subprocess.CalledProcessError as e:
            print(f"Error building Docker image: {e}")
            raise

    def purge(self, image_name: str) -> None:
        """
        Remove all containers and images associated with the given image name.

        Args:
            image_name: The name of the image to purge (without tag)
        """
        print(f"Purging all containers and images for {image_name}...")

        # Remove all containers using this image
        try:
            containers = self.client.containers.list(all=True)
            for container in containers:
                if any(image_name in tag for tag in container.image.tags):
                    print(f"Removing container {container.name}")
                    container.remove(force=True)

        except Exception as e:
            print(f"Error removing containers: {e}")

        # Remove all images with this name
        try:
            self.client.images.prune(filters={"dangling": False})
            images = self.client.images.list()
            for image in images:
                if any(image_name in tag for tag in image.tags):
                    print(f"Removing image {image.tags}")
                    self.client.images.remove(image.id, force=True)
        except Exception as e:
            print(f"Error removing images: {e}")


def main() -> None:
    # Register SIGINT handler
    # signal.signal(signal.SIGINT, handle_sigint)

    docker_manager = DockerManager()

    # Parameters
    image_name = "python"
    tag = "3.10-slim"
    # new_tag = "my-python"
    container_name = "my-python-container"
    command = "python -m http.server"
    running_container: RunningContainer | None = None

    try:
        # Step 1: Validate or download the image
        docker_manager.validate_or_download_image(image_name, tag, upgrade=True)

        # Step 2: Tag the image
        # docker_manager.tag_image(image_name, tag, new_tag)

        # Step 3: Run the container
        container = docker_manager.run_container_detached(
            image_name, tag, container_name, command
        )

        # Step 4: Attach and monitor the container logs
        running_container = docker_manager.attach_and_run(container)

        # Wait for keyboard interrupt
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping container...")
        if isinstance(running_container, RunningContainer):
            running_container.stop()
        container_or_none = docker_manager.get_container(container_name)
        if container_or_none is not None:
            docker_manager.suspend_container(container_or_none)
        else:
            warnings.warn(f"Container {container_name} not found.")

    try:
        # Suspend and resume the container
        container = docker_manager.get_container(container_name)
        assert container is not None, "Container not found."
        docker_manager.suspend_container(container)

        input("Press Enter to resume the container...")

        docker_manager.resume_container(container)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
