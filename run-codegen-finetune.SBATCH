#!/bin/bash
  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=200GB
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --job-name=codegen
#SBATCH --time=140:00:00
#SBATCH --mail-user=REPLACE_WITH_EMAIL_ID_WITHOUT_QUOTE
#SBATCH --mail-type=END

module purge

singularity exec --nv \
      --overlay overlay_image.ext3:ro \
      singularity_container.sif \
      /bin/bash -c "source /ext3/env.sh; conda activate name_of_the_conda_env; deepspeed --num_gpus 2 --num_nodes 1 run_clm.py --model_name_or_path=Salesforce/codegen-6B-multi --save_steps=100 --per_device_train_batch_size=1 --learning_rate 2e-5 --num_train_epochs 1 --output_dir=CodeGen/codegen-6B-verilog-3-epochs --report_to 'wandb' --dataset_name code_segments --tokenizer_name Salesforce/codegen-16B-multi --block_size 1024 --gradient_accumulation_steps 32 --do_train --do_eval --fp16 --overwrite_output_dir --deepspeed ds_config.json" > cat log_file.txt
