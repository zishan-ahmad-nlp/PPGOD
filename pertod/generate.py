import argparse
import json
import requests
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
import torch
from dataset import GPT21024Dataset 
from uttils import add_special_tokens, beam_search, generate_beam_sample, generate_sample, sample_seq, set_seed, top_k_top_p_filtering
                               
parser = argparse.ArgumentParser()
parser.add_argument("--device",default=torch.device('cpu'), required=False, help="torch.device object")
parser.add_argument("--root_dir",default='/Data/zishan/persuation/dataset/dataprocess1/simpletod_data', type=str, required=False, help="location of json dataset.")
parser.add_argument("--test_dir",default='/Data/zishan/persuation/dataset/dataprocess1/simpletod_data', type=str, required=False, help="location of test json dataset.")
parser.add_argument("--ids_file",default='./data/updated_data/ids.json', type=str, required=False, help="location of train, valid and test file indexes")
# parser.add_argument("--model", type=str, required=True, help="trained model location")
parser.add_argument("--model_dir",default='./weights/latest', type=str,  help="path to save trained model")
parser.add_argument("--num", default=100, type=int, required=False, help="number of predictions")
args = parser.parse_args()

# with open(args.ids_file,'r') as f:
#         js = json.load(f)
#         train_size = len(js['train_ids'])
#         valid_size = len(js['valid_ids'])
#         test_size = len(js['test_ids'])

model_file = "/Data/zishan/persuation/baselines/simpletod/output/gpt2/checkpoint-10000/pytorch_model.bin"
config_file = "/Data/zishan/persuation/baselines/simpletod/output/gpt2/checkpoint-10000/config.json"
model_checkpoint = "/Data/zishan/persuation/baselines/simpletod/output/gpt2/checkpoint-40000"

lines = None
with open(f'{args.root_dir}/test.lex') as f:
        lines = f.readlines()
test_size = len(lines)
# train_data = GPT21024Dataset(args.root_dir,args.test_dir,args.ids_file,mode='train',length=train_size)
# valid_data = GPT21024Dataset(args.root_dir,args.test_dir,args.ids_file,mode='valid',length=valid_size)
test_data = GPT21024Dataset(args.root_dir, model_checkpoint, mode='test',length=test_size)
# tokenizer = add_special_tokens()
# ignore_index = tokenizer.pad_token_id


# config = GPT2Config.from_json_file(config_file)
# model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
# state_dict = torch.load(model_file)
# model.load_state_dict(state_dict)
model.eval()
model.to(args.device)

# model = GPT2LMHeadModel.from_pretrained('gpt2')
# model.resize_token_embeddings(len(tokenizer))
# model.to(args.device)
# checkpoint = torch.load(model_file)
# model.load_state_dict(checkpoint)
generate_sample(test_data, tokenizer, model, model_checkpoint=model_checkpoint, num=test_size, length=120, temperature=0.7, top_k=20, top_p=0.9, device=args.device)
