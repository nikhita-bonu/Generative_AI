import torch

def sample_real_data(batch_size, mode="gaussian"):
    if mode == "gaussian":
        return torch.randn(batch_size, 2) * 1.2 + torch.tensor([2.0, 2.0])
    elif mode == "mixture":
        centers = torch.tensor([[2, 2], [-2, -2]])
        idx = torch.randint(0, 2, (batch_size,))
        return centers[idx] + 1.2 * torch.randn(batch_size, 2)
    #---------------------------------------------------------------
    # To-Do: Implement a real distribution

    elif mode == "mixture4":
        centers = torch.tensor([[2, 2], [2, -2], [-2, 2], [-2, -2]])
        idx = torch.randint(0, 4, (batch_size,))

        variance = 1.5

        return centers[idx] + variance * torch.randn(batch_size, 2)
    
    
    
    #---------------------------------------------------------------
    else:
        raise ValueError("Unknown mode")
