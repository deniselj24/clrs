# Adapted from run.py for the main CLRS dataset
# Pretrain a Gemma 2B model on CLRS text examples

import functools
import os
import shutil
from typing import Any, Dict, List, Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np
import json 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
# from evaluation import evaluate_text
import wandb
from datasets import load_dataset


# flags.DEFINE_list('algorithms', ['bfs', 'activity_selector', 'articulation_points', 'bellman_ford', 'binary_search', 'bridges', 'bubble_sort', 'dag_shortest_paths', 'dfs', 'dijkstra', 'find_maximum_subarray_kadane', 'floyd_warshall', 'graham_scan', 'heapsort', 'insertion_sort', 'jarvis_march', 'kmp_matcher', 'lcs_length', 'matrix_chain_order', 'minimum', 'mst_kruskal', 'mst_prim', 'naive_string_matcher', 'optimal_bst', 'quickselect', 'quicksort', 'segments_intersect', 'strongly_connected_components', 'task_scheduling', 'topological_sort'], 'Which algorithms to run.')
flags.DEFINE_list('algorithms', ['bfs'], 'Which algorithms to run.')
"""flags.DEFINE_list('train_lengths', ['4', '7', '11', '13', '16'],
                  'Which training sizes to use. A size of -1 means '
                  'use the benchmark dataset.')
flags.DEFINE_boolean('random_pos', True,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', True,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')"""
flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_integer('train_steps', 10000, 'Number of training iterations.')
flags.DEFINE_integer('eval_every', 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer('test_every', 500, 'Evaluation frequency (in steps).')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use.')
flags.DEFINE_float('grad_clip_max_norm', 1.0,
                   'Gradient clipping by norm. 0.0 disables grad clipping')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')

"""flags.DEFINE_enum('hint_repred_mode', 'soft', ['soft', 'hard', 'hard_on_eval'],
                  'How to process predicted hints when fed back as inputs.'
                  'In soft mode, we use softmaxes for categoricals, pointers '
                  'and mask_one, and sigmoids for masks. '
                  'In hard mode, we use argmax instead of softmax, and hard '
                  'thresholding of masks. '
                  'In hard_on_eval mode, soft mode is '
                  'used for training and hard mode is used for evaluation.')
flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')"""

flags.DEFINE_string('dataset_path', './data_with_trace',
                    'Path in which training data is stored.')

FLAGS = flags.FLAGS

"""class CustomTrainer(Trainer):
  def eval_loss(self, model, inputs, return_outputs=False):
    if self.eval:
      # Custom loss based on string matching where the target is checked to be a substring of the generated text
      generated_ids = model.generate(
        inputs["input_ids"],
        max_length=8192,
        pad_token_id=self.tokenizer.pad_token_id,
        eos_token_id=self.tokenizer.eos_token_id,
      )
      generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
      target_text = self.tokenizer.batch_decode(inputs["labels"], skip_special_tokens=True)
            
      correct = sum(1 for gen, tgt in zip(generated_text, target_text) if tgt.strip() in gen.strip())
      loss = 1 - (correct / len(generated_text))

    return (loss, outputs) if return_outputs else loss"""


def main(unused_argv):
  # Gemma 2B base model
  model_config = AutoConfig.from_pretrained("./configs/gemma-2b.json")
  train_model = AutoModelForCausalLM.from_config(model_config)
  tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

  # create dataset -- traces were previously called hints in the original CLRS benchmark
  # traces are in contained the target field of the json
  train_file = './training_data/train.json'
  # train_examples = {"prompt": [], "targets": [], "trace": []}
  train_examples = {"trace": []}
  for algorithm in FLAGS.algorithms:
    data_path = f"./data/train/{algorithm}.json"
    with open(data_path, 'r') as f:
      text_data = json.load(f)
      # train_examples["prompt"].extend([example["prompt"] for example in text_data["examples"]])
      # train_examples["targets"].extend([example["references"][0] for example in text_data["examples"]])
      train_examples["trace"].extend([example["prompt"] + tokenizer.sep_token + example["references"][0] for example in text_data["examples"]])
  
  val_examples = {"trace": []}
  val_targets = {"trace": []}
  for algorithm in FLAGS.algorithms:
    data_path = f"./data/val/{algorithm}.json"
    with open(data_path, 'r') as f:
      text_data = json.load(f)
      val_examples["trace"].extend([example["prompt"] for example in text_data["examples"]])
      val_targets["trace"].extend([example["references"][0] for example in text_data["examples"]])
  val_file = './training_data/val.json'
  val_target_file = './training_data/val_target.json'

  with open(train_file, 'w') as f:
      json.dump(train_examples, f)
  with open(val_file, 'w') as f:
    json.dump(val_examples, f)
  with open(val_target_file, 'w') as f:
      json.dump(val_targets, f)
  hf_datasets = load_dataset(
    "json",
    data_files={
      "train": train_file,
      "val": val_file,
      "val_target": val_target_file,
    },
  )

  with open("./configs/train.json", "r") as f:
    train_config = json.load(f)

  # init wandb 
  user = "deniselj24"
  project = "clrs-text"
  display_name = "gemma-2-2b"
  wandb.init(entity=user, project=project, name=display_name)

  # tokenize dataset 
  def tokenize(examples):
    # text = [tokenizer.bos_token + examples["inputs"][i].strip() + tokenizer.sep_token + examples["targets"][i].strip() + tokenizer.eos_token
            # for i in range(len(examples["inputs"]))
            # ]
    text = [tokenizer.bos_token + examples["trace"][i].strip() + tokenizer.eos_token for i in range(len(examples["trace"]))]
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        # context length for Gemma 2b
        max_length=8192
    )
  
  tokenized_dataset = hf_datasets.map(
    tokenize,
    batched=True,
    remove_columns=hf_datasets.column_names
  )
  data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
  
  # train model 
  training_args = TrainingArguments(
    seed=FLAGS.seed,
    per_device_train_batch_size=train_config["per_device_train_batch_size"],
    per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
    learning_rate=train_config["learning_rate"], 
    num_train_epochs=train_config["num_train_epochs"],
    weight_decay=train_config["weight_decay"],
    logging_steps=train_config["logging_steps"],
    evaluation_strategy="steps",
    eval_steps=train_config["eval_steps"],
    save_steps=train_config["save_steps"],
    # save_total_limit=train_config["save_total_limit"],
    load_best_model_at_end=train_config["load_best_model_at_end"],
    output_dir=train_config["output_dir"],
    **train_config
  )

  trainer = Trainer(
    model=train_model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset={
      "valid": tokenized_dataset["val"],
      "valid_target": tokenized_dataset["val_target"],
    },
  )
  trainer.train()


if __name__ == '__main__':
  app.run(main)
