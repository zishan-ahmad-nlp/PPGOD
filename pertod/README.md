


# PerTOD: A GPT based language model for persuasive dialogue generation

 

## Installation

The package general requirements are

- Python >= 3.6
- Pytorch >= 1.2 (installation instructions [here](https://pytorch.org/))
- Transformers >= 2.5.1 (installation instructions [here](https://huggingface.co/transformers/))
 
1- The package can be installed by running the following command.  

```pip install -r requirements.txt```

To train PerTOD on the sequence of persona+context+intent. 

```
train_end2end_per.sh $GPU gpt2 $GPT2_TYPE $BATCH
```

 
### Generation:

Set the checkpoint and test file in the file generate.py and run it
```
CUDA_VISIBLE_DEVICES=$GPU python generate.py
```



