import json


# from vllm import LLM, SamplingParams

class model_generator:
    def __init__(self, is_few_shot, device, is_vllm, model_path, few_shot_path):
        '''
        Args:
            f_path: The path for original data file in json format
            is_few_shot: Whether to utilize few-shot setting
            device: device number of cuda
            model_path: The path of the model
            few_shot_path: The path for few-shot data file
        '''
        self.model_path = model_path
        self.is_few_shot = is_few_shot
        if is_few_shot and few_shot_path == None:
            raise ValueError("Cannot find few-shot path")
        self.few_shot_path = few_shot_path
        self.device = device
        self.is_vllm = is_vllm

    def model_init_vllm(self):
        '''
        Init the model using vllm
        '''
        model = LLM(
            model=self.model_path,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True
        )
        tokenizer = model.get_tokenizer()
        return model, tokenizer

