from typing import Tuple

from transformers import PreTrainedTokenizerFast


def get_tokenizer(tokenizer_file_path: str) -> Tuple[PreTrainedTokenizerFast, str]:
    
    """
    Load a custom pretrained tokenizer. 
    
    Args: tokenizer_file_path
        File path for a pretrained huggingface tokenizer.
    
    Returns:
        Huggingface PreTrainedTokenizerFast tokenizer object and blank token for CTC loss.
    
    """
    try: ##Check if tokenizer is defined
        tokenizer
    except NameError as e: ## If tokenizer is not defined then initialize it
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file_path)
        
    finally:
        special_tokens_dict = {'pad_token': '[PAD]',
                           'sep_token': '[SEP]',
                           'mask_token': '[MASK]'}

        tokenizer.add_special_tokens(special_tokens_dict)

        blank_token = "[BLANK]"
    
    return tokenizer, blank_token