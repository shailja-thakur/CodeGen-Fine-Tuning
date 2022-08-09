# CodeGen fine tuning on a local server

## This is a step by step process for fine-tuning CodeGen language models on specific programming languages uisng huggingface transformers and deepspeed

CodeGen is a suite of code based language models by SalesForce (https://github.com/salesforce/CodeGen/blob/main/README.md). Model sizes vary with respect to their training corpus, and model parameters. Models are named as per the convention codegen-{model-size}-{data}.

`model-size` has 4 options: `350M`, `2B`, `6B`, `16B`, which represent the number of parameters in each model.

`data` has 3 options: `nl`, `multi`, `mono`.

* `nl` models are randomly initialized and trained on [The Pile](https://github.com/EleutherAI/the-pile), a 825.18 GB English text corpus.
* `multi` models are initialized from `nl` models and then trained on a corpus with code data consisting of multiple programming languages.
* `mono` models are initialized from `multi` models and then trained on a corpus with Python code data.

A detailed description of the models isas follows:

## CodeGen models

| model name | data | model-size|
| ------ | ----------- |--------|
| codegen-350M-nl   | nl |350M|
| codegen-350M-multi  | multi | 350M |
| codegen-350M-mono    | mono| 350M|
|codegen-2B-nl       | nl| 2B|
|codegen-2B-multi      |multi| 2B|
|codegen-2B-mono       |mono| 2B|
|codegen-6B-nl       |nl|6B|
|codegen-6B-multi      |multi|6B|
|codegen-6B-mono       |mono|6B|
|codegen-16B-nl      |nl|16B|
|codegen-16B-multi       |multi|16B|
|codegen-16B-mono      |mono|16B|



### Following is a detailed set of instruction for replicating the CodeGen fine-tuning on a local server:

The following steps have been tested on an HPC with a sungularity container with Ubuntu20.04 and 50GB RAM. However, the setup can also be replicated on a machine with ubuntu 20.04.

#### Prepare training corpus. 

For CodeGen models, the data format has to be in a loose json format with one json per line followed by a new line as follows:

`{‘text’: your data chunk 1}\n`
`{‘text’: your data chunk 2}\n`
`...`
 
I used the following code snippet to prepare the json,

```python
with open('code_segments.json','a') as f:
    for row in df_code['text'].values:
        dic={"text":str(row)}
        ob=json.dumps(dic)
        f.write(ob)
        f.write('\n')
f.close()
```
Note, in this case, the for loop iterates a pandas dataframe `df_code` with a column named `text`. You may tweak the code snippet according to the type of data you will be rading.

#### Prepare the environment on your machine 

I recommend creating a conda environment for fine-tuning. I created a conda environment inside the singularity container, however, if you are not using container, you may create a conda environment direclty on your machine,

```
conda create --name anyname python=3.X
then, activate the environment
conda activate anyname
```

And later, install the following software libraries inside the environment (`conda activate name_of_the_conda_env`)

+ Clone the transformers repo from GitHub
```git clone https://github.com/huggingface/transformers```

+ And navigate to the path `YOUR_ROOT/transformers/examples/pytorch/language-modeling/`
+ Run the sequence of pip as follows to install the requirements
```
pip install -r requirements.txt
```

```
pip install git+https://github.com/huggingface/transformers/
```
```
pip install deepspeed
```

+ Put the json file we prepared in teh first step in a folder on the path as above (../transformers/examples/pytorch/language-modeling/), and the name of the folder should be the same as the name of your json file without extension.
+ At this point you are ready to run fine-tuning if everything is good — it is possible that you run into some package conflicts and other issues, which you will have to resolve along the way, you can also let me know, perhaps I must have already encountered those issues

+ At this point, you are ready to run the fine-tuning. The following command runs fine-tuning script `run_clm.py` using deepspeed (https://huggingface.co/docs/transformers/main_classes/deepspeed). In this case, deepspeed request two gpus on a node. You can play around with the `run_clm.py` options and deepspeed configuration (~ds_config.json`) and change the save_steps, model name, number of epochs to train, input token length, and otehr parametrs. The following configuration of `run_clm` has been tested to work on teh HPC wit ubuntu 20.04.

```
deepspeed --num_gpus 2 --num_nodes 1 run_clm.py --model_name_or_path=Salesforce/codegen-6B-multi --save_steps=100 --per_device_train_batch_size=1 --learning_rate 2e-5 --num_train_epochs 1 --output_dir=CodeGen/codegen-6B-verilog-3-epochs --report_to 'wandb' --dataset_name code_segments_verilog --tokenizer_name Salesforce/codegen-16B-multi --block_size 1024 --gradient_accumulation_steps 32 --do_train --do_eval --fp16 --overwrite_output_dir --deepspeed ds_config.json"
```

To run the fine-tuning as a job on HPC, I created a slurm script (`run-codegen-finetune.SBATCH`) which runs the above command in a slurm script with conda environment within singularity container. 


+ The deepspeed configurations is included in the ds_config.json file
+ One more step, if you look at the arguments of the `run_clm.py` script above, you will notice that there is a term “wandb”. It is similar to `tensorboard`. `wandb` is a [web portal] (https://wandb.ai/) that is integrated with the transformers, and helps visualize the system usage, logs, and other details while the fine-tuning progress.
+ Make sure that you install `wandb` as `pip install wandb` and register on their portal 
+ Next, log in to wandb in the terminal within your singulartity container (or, in the terminal on your machine) before executing the deepspeed command above (or, running the slurm script) like,

```wandb login```

Note, that the wandb session may timeout, so, you can also open a new termina, login to wandb, and leave that terminal open while you execute the fine-tuning in another window.

+ Upon execution of the wandb login command, the following prompt on the terminal will ask you to paste the API key (available from your profile page on the [wandb portal] (https://wandb.ai)),
 
It is possible to remove the wandb option from the fine-tuning altogether by removing the option and continue fine-tuning. If you would like to use tensorboard in place of wandb, then simply replace the wandb with tensorboard, and configure tensorboard path (https://www.tensorflow.org/tensorboard/get_started).

+ Run the above command (or, start the batch job), check your job log file for any error
+ If you are running the fine-tuning on HPC, at first, I would suggest you request only one GPU on one node with lesser memory, which will be allocated easily, and you can resolve any error that pops up along the way.
+ If everything is installed and compatible, the fine-tuning should execute and you will be able to track the progress on wandb portal and from the log file on your machine.

## Next, how to evaluate the fine-tuned model coming soon .. 
