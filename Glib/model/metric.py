import torch


def mse_metric(output, target):
    with torch.no_grad():
        assert output.shape[0] == len(target)
        accuracy = torch.mean((output - target)**2, dim=0)
        if accuracy.device.type == 'cuda':
            accuracy = accuracy.cpu()
    return accuracy.numpy()

def mae_metric(output, target):
    with torch.no_grad():
        assert output.shape[0] == len(target)
        accuracy = torch.mean(torch.abs(output-target), dim=0)
        if accuracy.device.type == 'cuda':
            accuracy = accuracy.cpu()
    return accuracy.numpy()
