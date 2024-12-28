import torch
import matplotlib.pyplot as plt
import numpy as np

def get_device():
    SEED = 1 # Seed is to generate the same random data for each run
    # For reproducibility
    torch.manual_seed(SEED)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}\n")
        torch.cuda.manual_seed(SEED)
    
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    else:
        torch.mps.manual_seed(SEED)

    return device

def transform_data_to_numpy(dataset):
    exp_data = torch.tensor(dataset.data) / 255.0  # Convert to tensor and normalize
    print('[Train]')
    print(' - Numpy Shape:', exp_data.cpu().numpy().shape)
    print(' - Tensor Shape:', exp_data.size())
    print(' - min:', exp_data.min())
    print(' - max:', exp_data.max())
    print(' - mean by channel:', torch.mean(exp_data, (0,1,2)))
    print(' - std by channel:', torch.std(exp_data, (0,1,2)))
    print(' - mean (overall):', exp_data.mean())
    print(' - std (overall):', exp_data.std())
    print(' - var:', torch.var(exp_data, (0,1,2)))
        
    return exp_data

def printSampleImages(dataset):
    iter_data = iter(dataset)
    image, label = next(iter_data)
    plt.figure
    plt.imshow(image[0].numpy().squeeze(), cmap='gray_r')
    plt.figure
    plt.imshow(np.transpose(image, (1, 2, 0)), interpolation='nearest')


    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        image, label = dataset[index]
        plt.subplot(6, 10, index)
        plt.axis('off')
        # plt.imshow(image[0].numpy().squeeze(), cmap='gray_r')
        plt.imshow(np.transpose(image, (1, 2, 0)), interpolation='nearest')