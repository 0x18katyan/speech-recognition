# Work in Progress

Repository for Automatic Speech Recognition. Repository uses [Mozilla CommonVoice English](https://commonvoice.mozilla.org/) dataset.

Refer to [Notes](https://github.com/0x18katyan/Notes/tree/master/Speech%20Recognition) for moderate references.

- **CTC** isn't numerically stable so we can't we mixed precision training.

- **Do not use high large output space with CTC.***

    1. It will make it difficult for the algorithm to align.

    2. The gradients will be huge and more computation will be required.

## TODO

- [x] DataLoading for CommonVoice dataset.
- [ ] Write Encoder-Decoder Model.
- [ ] Perform Loss alignment.
- [ ] Write Training Loop
- [ ] Tensorboard Logging
- [ ] Inference Code

## **Doesn't Work**
