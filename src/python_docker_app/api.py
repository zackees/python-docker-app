class DockerApp:
    def __init__(self, image_name: str, container_name: str) -> None:
        self.image_name = image_name
        self.container_name = container_name
        pass

    def pull(self) -> None:
        """Pull the Docker image."""
        # Here you would implement the logic to pull the Docker image
        # using the Docker SDK for Python or any other method you prefer.
        pass

    def run(self) -> None:
        """Run the Docker container."""
        # Here you would implement the logic to run the Docker container
        # using the Docker SDK for Python or any other method you prefer.
        pass
