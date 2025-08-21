import torch


def get_device(device: str | torch.device | None = None) -> torch.device:
    """
    Get the appropriate device for PyTorch based on the input string.
    If no device is specified, it will automatically select 'cuda' if available,
    otherwise 'mps' for Apple Silicon, or 'cpu'.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # If device is already a torch.device, return it as is
    if isinstance(device, torch.device):
        return device

    return torch.device(device)
