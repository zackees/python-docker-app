"""
Example script demonstrating how to use DockerApp to run a simple web server.
"""

import sys
import time
from pathlib import Path

from python_docker_app.api import DockerApp
from python_docker_app.volume import Volume


def run_web_server():
    """Run a simple Python web server in a Docker container."""
    try:
        # Create a DockerApp instance for a Python web server
        app = DockerApp(
            image_name="python",
            container_name="python-web-server-example",
            tag="3.10-slim",
        )

        # Pull the image (don't force upgrade to avoid unnecessary downloads)
        print("Pulling Python image...")
        app.pull(upgrade=False)

        # Create a volume to serve local files
        current_dir = Path.cwd()
        volumes = [Volume(str(current_dir), "/app", "ro")]

        # Run the container with a simple HTTP server
        print("Starting web server...")
        app.run(
            command="python -m http.server 8080 --directory /app",
            volumes=volumes,
            ports={8080: 8080},  # Map host port 8080 to container port 8080
            detached=True,
            remove_previous=True,
        )

        # Attach to the container to monitor logs
        running_container = app.attach()
        assert running_container is not None, "Failed to attach to the container"

        print("\nWeb server is running at http://localhost:8080")
        print("Serving files from:", current_dir)
        print("Press Ctrl+C to stop the server...")

        # Wait for keyboard interrupt
        try:
            while app.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping web server...")

        # Stop the container
        app.stop()
        print("Web server stopped.")

    except Exception as e:
        print(f"An error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_web_server())
