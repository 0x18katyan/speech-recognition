from typing import List, Tuple, Union, Optional
from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import Whitespace, Digits, Sequence
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase

import pandas as pd
from pandas import Series


def train_tokenizer(tokenizer_file_path: str, 
                    text: Union[Series, List[str]], 
                    special_tokens: Optional[List[str]], 
                    vocab_size: Optional[int] = 50, 
                    min_frequency: Optional[int] = 5) -> PreTrainedTokenizerFast:
    
    """Trains a Tokenizer on given text.
    
    Args:
        tokenizer_file_path (str): path for saving the tokenizer.

    Returns:
        PreTrainedTokenizerFast: an instance of a HuggingFace PreTrainedTokenizer
    """
    
    special_tokens= ["[BLANK]", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size = vocab_size, min_frequency = min_frequency, special_tokens = special_tokens)

    normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True)])
    
    tokenizer.post_processor = TemplateProcessing(single="[BOS] $A [EOS]",
                                                  special_tokens=[
                                                      ("[BOS]", tokenizer.token_to_id("[BOS]")),
                                                      ("[EOS]", tokenizer.token_to_id("[EOS]")),
                                                      ]
                                                  )
    
    tokenizer.train_from_iterator(iterator = text, trainer=trainer)
    
    tokenizer.save(tokenizer_file_path)
    
    return tokenizer

def get_tokenizer(tokenizer_file_path: str) -> PreTrainedTokenizerFast:
    
    """
    Load a custom pretrained tokenizer. 
    
    Args: tokenizer_file_path
        File path for a pretrained huggingface tokenizer.
    
    Returns:
        Huggingface PreTrainedTokenizerFast tokenizer object.
    """
    
    try: ##Check if tokenizer is ~defined~ saved else train and save
        tokenizer
    except NameError as e: ## If tokenizer is not defined then initialize it
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file_path)
    finally:
        special_tokens_dict = {'bos_token': '[BOS]',
                               'pad_token': '[PAD]',
                               'sep_token': '[SEP]',
                               'mask_token': '[MASK]'}

        tokenizer.add_special_tokens(special_tokens_dict)
    
    return tokenizer