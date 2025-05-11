class DockerApp:
    def __init__(self, image_name: str, container_name: str) -> None:
        self.image_name = image_name
        self.container_name = container_name
