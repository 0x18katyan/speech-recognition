import torch
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def plot_waveform(waveform, sampling_rate, title = "Waveform", xlim = None, ylim = None):
    
    waveform = waveform.numpy()
    
    if len(waveform.shape) > 2: ## Remove the batch dimension
        waveform = waveform[0]
    
    num_channels, num_frames = waveform.shape
    
    time_axis = torch.arange(0, num_frames) / sampling_rate
    
    # num_channels is for number of columns and 1 for number of rows; we need 2 columns for each
    # of the channels and a shared row for the time axis.
    
    figure, axes = plt.subplots(num_channels, 1)
    
    ## For ensuring that it loops over the axes
    if num_channels == 1:
        axes = [axes]
    
    ## Range over the channel dimension.
    for c in range(num_channels):
        
        ## Plot the waveform against the time dimension
        axes[c].plot(time_axis, waveform[c], linewidth = 1)
        ## Show the grid
        axes[c].grid(True)
        
        ## Set the y_label for the channels
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        
        ## Set the x and y limits.
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    
    figure.suptitle(title)
    plt.show(block=False)

    
def play_audio(signal: torch.Tensor, sampling_rate: torch.Tensor):
    """
    Displays a widget for playing an audio.
    
    Args:
        signal: signal of audio
        sampling_rate: sampling_rate of audio
    """
    signal = signal.cpu().numpy()
    
    num_channels, num_frames = signal.shape
    
    if num_channels == 1:
        display(Audio(data = signal[0], rate = sampling_rate))
    elif num_channels == 2:
        display(Audio(data = (signal[0], signal[1]), rate = sampling_rate))
    else:
        raise ValueError("Signal has more than 2 channels which is not supported.")
    