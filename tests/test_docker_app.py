"""
Unit test file.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from python_docker_app.api import DockerApp
from python_docker_app.volume import Volume


class DockerAppTester(unittest.TestCase):
    """Main tester class."""

    def setUp(self) -> None:
        """Set up test environment."""
        # Skip tests if CI environment or no Docker
        if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
            self.skipTest("Skipping Docker tests in CI environment")

        # Use a test-specific image and container name
        self.image_name = "python"
        self.container_name = "python-docker-app-test"
        self.tag = "3.10-slim"

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Try to stop any containers that might be running
        try:
            app = DockerApp(self.image_name, self.container_name, self.tag)
            if app.is_running():
                app.stop()
        except Exception:
            pass

    def test_simple_image(self) -> None:
        """Test basic DockerApp functionality."""
        # Create a DockerApp instance
        app = DockerApp(self.image_name, self.container_name, self.tag)

        # Test pulling the image
        # We don't force upgrade to avoid unnecessary downloads in tests
        pulled = app.pull(upgrade=False)
        self.assertIsNotNone(pulled, "Image pull should not be None")
        # pulled could be True or False depending on if image was already present

        # Test running the container with a simple command
        container = app.run(
            command="python -c 'print(\"Hello from Docker!\")'",
            detached=True,
            remove_previous=True,
        )

        # Verify container was created
        self.assertIsNotNone(container, "Container should not be None")

        # Check if container is running
        is_running = app.is_running()
        self.assertTrue(is_running, "Container should be running")

        # Stop the container
        app.stop()

        # Verify container is stopped
        is_running = app.is_running()
        self.assertFalse(is_running, "Container should be stopped")

    @patch("python_docker_app.docker_manager.DockerManager")
    def test_docker_app_mock(self, mock_docker_manager_class) -> None:
        """Test DockerApp with mocked DockerManager."""
        # Setup mocks
        mock_docker_manager = MagicMock()
        mock_docker_manager_class.return_value = mock_docker_manager
        mock_docker_manager.validate_or_download_image.return_value = True
        mock_container = MagicMock()
        mock_docker_manager.run_container_detached.return_value = mock_container
        mock_docker_manager.is_container_running.return_value = True

        # Create DockerApp with mocked DockerManager
        app = DockerApp("test-image", "test-container")

        # Test pull
        result = app.pull(upgrade=True)
        self.assertTrue(result)
        mock_docker_manager.validate_or_download_image.assert_called_once_with(
            "test-image", "latest", upgrade=True
        )

        # Test run
        volumes = [Volume("/host/path", "/container/path", "rw")]
        ports = {8080: 80}
        container = app.run(
            command="test-command", volumes=volumes, ports=ports, detached=True
        )
        self.assertEqual(container, mock_container)
        mock_docker_manager.run_container_detached.assert_called_once()

        # Test is_running
        is_running = app.is_running()
        self.assertTrue(is_running)
        mock_docker_manager.is_container_running.assert_called_once_with(
            "test-container"
        )

        # Test stop
        app.stop()
        mock_docker_manager.suspend_container.assert_called_once()


if __name__ == "__main__":
    unittest.main()
