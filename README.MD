# Work in Progress

Repository for Automatic Speech Recognition. Repository uses [Mozilla CommonVoice English](https://commonvoice.mozilla.org/) dataset.

Refer to [Notes](https://github.com/0x18katyan/Notes/tree/master/Speech%20Recognition) for moderate references.

- **CTC** isn't numerically stable so we can't we mixed precision training.

- **Do not use high large output space with CTC.***

    1. It will make it difficult for the algorithm to align.

    2. The gradients will be huge and more computation will be required.

## TODO

- [x] DataLoading for CommonVoice dataset.
- [x] Write Encoder-Decoder Model.
- [ ] Add Transducer Model: RNN Language Model.
- [ ] Train AutoRegressive Transducer Language Model on the Common Voice train data.
- [ ] Calculate Cross Entropy Loss over the outputs.
- [x] Calculate CTC Loss. (Obsolete: Conformer Paper doesn't use CTC, it is not mentioned anywere in the paper.)
- [x] Write Training Loop
- [x] Tensorboard Logging: Added Gradient Flow Logging and Loss Logging. 
- [ ] PyTorch Lightning Training?
- [ ] Inference Code

## **Doesn't Work in Current State**
