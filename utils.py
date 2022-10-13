import torch


def get_device():
    # return torch.device("mps" if torch.backends.mps.is_built() else 'cpu')
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, str):
        return data
    return data.to(device, non_blocking=True)
