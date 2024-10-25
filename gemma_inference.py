# Choose variant and machine type
VARIANT = '2b-it' #@param ['2b', '2b-it', '9b', '9b-it', '27b', '27b-it']
#VARIANT = '7b'
MACHINE_TYPE = 'cuda' #@param ['cuda', 'cpu']

CONFIG = VARIANT[:2]
if CONFIG == '2b':
  CONFIG = '2b-v2'

weights_dir = "/home/compu/.cache/kagglehub/models/google/gemma-2/pyTorch/gemma-2-2b-it/1"
#weights_dir = "/home/compu/.cache/huggingface/hub/models--google--gemma-7b-it-pytorch/snapshots/model_weights"

import os
# Ensure that the tokenizer is present
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

# Ensure that the checkpoint is present
ckpt_path = os.path.join(weights_dir, f'model.ckpt')
assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

from gemma_pytorch.gemma.config import get_model_config
from gemma_pytorch.gemma.model import GemmaForCausalLM

import torch, time

model_config = get_model_config(CONFIG)
model_config.tokenizer = tokenizer_path
model_config.quant = 'quant' in VARIANT

# Instantiate the model and load the weights.
torch.set_default_dtype(model_config.get_dtype())
device = torch.device(MACHINE_TYPE)
model = GemmaForCausalLM(model_config)
model.load_weights(ckpt_path)
model = model.to(device).eval()

# ANSI escape codes for text colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

# Chat templates
USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn><eos>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn><eos>\n"

def chat_with_model(prompt):
    full_prompt = USER_CHAT_TEMPLATE.format(prompt=prompt) + '<start_of_turn>model\n'
    start_time = time.time()
    results = model.generate(
        full_prompt,
        device=device,
        output_len=256,
    )
    end_time = time.time()

    print(f"{GREEN}Answer : {RESET}{results}")
    print(f"{BLUE}Latency (execution time) : {RESET}{end_time - start_time:.2f} seconds")

while True:
    user_input = input(f"{YELLOW}User : {RESET}")
    if user_input.lower() in ['exit', 'quit']:
        print(f"{RED}대화를 종료합니다.{RESET}")
        break
    chat_with_model(user_input)