import torch
import torch.nn.functional as F

class BatchUtils:
    
    @staticmethod
    def pad(tensor: torch.Tensor, target_length: int):
        """
        Pads tensor according to the given length.
        
        Args:
            tensor: torch.Tensor
                tensor to be padded
        
            out_length: int
                target length of tensor
        
        Returns:
            tensor: torch.tensor
                padded tensor
        """
        
        length = target_length - tensor.shape[-1]

        if length <= 0:
            return tensor
        else:
            return F.pad(tensor, (0, length), "constant", 0)
    
    @staticmethod
    def collate_fn(batch):
    
        waveforms = [sample['waveform'] for sample in batch]
        
        sentences = [sample['sentence'] for sample in batch]
        
        mel_specs = [sample['mel_spec'] for sample in batch]
                
        max_len_waveform = max([waveform.shape[1] for waveform in waveforms])
        
        max_len_specs = max([mel_spec.shape[-1] for mel_spec in mel_specs])

        padded_sentences = tokenizer(sentences, padding=True, return_tensors = 'pt', return_attention_mask=False)

        padded_waveforms = torch.stack([BatchUtils.pad(waveform, max_len_waveform) for waveform in waveforms])
        
        padded_mel_specs = torch.stack([BatchUtils.pad(mel_spec, max_len_specs) for mel_spec in mel_specs])
        
        return {"waveforms": padded_waveforms, "sentences": padded_sentences['input_ids'], "mel_specs": padded_mel_specs}