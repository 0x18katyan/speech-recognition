from transformers import AutoModelForCausalLM

def get_model(model_path: str):
    
    lmModel = AutoModelForCausalLM.from_pretrained(model_path, is_decoder = True)
    
    return lmModel
