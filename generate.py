import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, HfArgumentParser
from configs import ModelArguments, GenerationArguments


device = "cuda" if torch.cuda.is_available() else "cpu"
parser = HfArgumentParser((ModelArguments, GenerationArguments))
model_args, generation_args = parser.parse_args_into_dataclasses()


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def stopping_criteria(tokenizer, stop_words):
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria


tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_args.pretrained_model_name_or_path, device_map="auto")
stopping = stopping_criteria(tokenizer, ["\n\nHuman:"])

inputs = tokenizer("\n\nHuman: Hi, can you give me ideas on what to do during the weekend?\n\nAssistant:", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
out = model.generate(**inputs, stopping_criteria=stopping, **vars(generation_args))
print(tokenizer.batch_decode(out))
