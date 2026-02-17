from dataclasses import dataclass


@dataclass
class MofaConfig:
    texture_size: int = 1024
    device: str = "cpu"
    output_format: str = "glb"
    export_identity: bool = True
    image_size: int = 224
    beta_dim: int = 128
