import os
import json
from tuner.utils.config import args
import math
from scipy.stats import poisson
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
)
from tuner.data.data_manager import DataManager
from tuner.evaluate.w_retrieval import infer_evaluate
from tuner.utils.answer_processor import generate_answer_w_noisy_retrieval,generate_answer_w_two_retrieval, generate_answer_w_one_retrieval
from trl import PreTrainedModelWrapper
from tuner.utils.select_retrieve import  rerank_two_ctx
import random

record_list = []
class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Linear(hidden_size, 4)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output


class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    r"""
    An autoregressive model with a value head in addition to the language model head.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, `push_to_hub` and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the `ValueHead` class. Currently, the supported args are:
            - **summary_dropout_prob** (`float`, `optional`, defaults to `None`) -- The dropout probability for the
                `ValueHead` class.
            - **v_head_initializer_range** (`float`, `optional`, defaults to `0.2`) -- The initializer range for the
                `ValueHead` if a specific initialization strategy is selected.
            - **v_head_init_strategy** (`str`, `optional`, defaults to `None`) -- The initialization strategy for the
                `ValueHead`. Currently, the supported strategies are:
                - **`None`** -- Initializes the weights of the `ValueHead` with a random distribution. This is the default
                    strategy.
                - **"normal"** -- Initializes the weights of the `ValueHead` with a normal distribution.

    """
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)

        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the value head. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state).squeeze(-1)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model.
        Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
        method of the wrapped model for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, "v_head", self.v_head)

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]

            self.v_head = self.v_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class adaProcessManager():
    def __init__(
        self,
        accelerator,
        model_path = args.model_name_or_path,
    ):
        self.accelerator = accelerator
        self.model_path = model_path
        self.sim = Similarity(temp=1.0)
        
        self.model_config = AutoConfig.from_pretrained(self.model_path)
        self.data_manager = DataManager(
                self.model_config,
                args.training_stage_num,
                self.model_path
            )
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_path,config=self.model_config)
        self.model.pretrained_model.resize_token_embeddings(len(self.data_manager.tokenizer))
        
        self.logger = get_logger(__name__)

    
    def ranking_loss(self, emb,min_index,max_index,batch_size,score_distance):
        '''
        Regularization
        '''
        ret_score = torch.abs(score_distance[:,min_index]-score_distance[:,max_index])**2
        return ret_score.view(batch_size*1)
    
    def compute_loss(self, model, batch, print_loss):
        """
        batch = [batch, training_stage, seq_len]
        implementation of L_raat
        """
        batch_size = batch["labels"].shape[0]
        temp_training_stage = batch["labels"].shape[1]
        sub_batches = [{key: batch[key][:,time,:] for key in ["input_ids", "attention_mask"]} for time in range(temp_training_stage)]
        
        score_list = []
        suffix_mask_list = []
        embeds = []
        logits_list = []
        mask_list = []
        cls_list = []
        for batch_index, sub_batch in enumerate(sub_batches):
            #local_outputs = model(**sub_batch, output_hidden_states=True, return_dict=True)
            lm_logits, loss, value = model(**sub_batch)
            cls_logits = value[:,-1]
            cls_list.append(cls_logits)
            #embeds.append(embedding)
            local_logits = lm_logits
            local_mask = sub_batch["attention_mask"] & (~batch["prefix_mask"][:, batch_index, :]) #[batch, seq_len]
            local_labels = batch["labels"][:, batch_index, :]

            shift_logits = local_logits[..., :-1, :].contiguous() #[batch, seq_len-1, token_num]
            #logits_list.append(shift_logits)
            shift_logits = F.log_softmax(shift_logits, dim=2) #[batch, seq_len-1, token_num]
            shift_masks = local_mask[..., :-1] #[batch, seq_len-1]
            #mask_list.append(shift_masks)
            shift_labels = local_labels[..., 1:].view(batch_size, -1, 1) #[batch, seq_len-1, 1]

            selected_logits = torch.gather(input=shift_logits, dim=2, index=shift_labels).view(batch_size, -1) #[batch, seq_len-1]
            selected_logits[shift_masks != 1] = 0.0 #[batch, seq_len-1]
            sentence_logits = torch.sum(selected_logits, dim=1) #[batch]
            sentence_logits = sentence_logits.view(batch_size, 1)
            score_list.append(sentence_logits)
            suffix_mask_list.append(torch.sum(shift_masks, dim=1).view(batch_size, 1))

        # Incorporating Noise Awareness
        cls_cat = torch.cat(cls_list, dim=0)
        targets = torch.tensor([0, 1, 2, 3]).cuda()
        cls_criterion = nn.CrossEntropyLoss()
        cls_loss = cls_criterion(cls_cat, targets)

        sum_scores = torch.cat(score_list, dim=1) #[batch, training_stage]
        suffix_mask = torch.cat(suffix_mask_list, dim=1) #[batch, training_stage]
        scores = sum_scores / suffix_mask #[batch, training_stage]

        total_loss = 0
        ada_score = sum_scores 
        _, min_index = ada_score[:,:].min(dim=1)
        _, max_index = ada_score[:,:].max(dim=1)
        # Regularization
        reg_loss = self.ranking_loss(embeds,min_index,max_index,batch_size,sum_scores)
        record_list.append(min_index)
        print('============================')
        self.logger.info('need to check:'+ str(min_index.item()),main_process_only=False)
        print('need to check:'+ str(min_index.item()))

        total_loss += args.reg_weight*reg_loss 
        print_loss[0].append(reg_loss.item())
        sft_index = batch["sft_index"].view(batch_size, 1)
        # Model prioritizes the selection of the largest loss to guide subsequent parameter update
        min_scores= sum_scores[:,min_index]
        sft_scores = min_scores.view(batch_size*1)
        sft_loss = torch.mean(-sft_scores).to(local_logits.dtype)
        sft_loss = args.sft_weight * sft_loss
        total_loss += sft_loss
        total_loss += cls_loss 
        print('cls:{cls}'.format(cls=cls_loss.item()))
        print('sft:{sft}'.format(sft=sft_loss.item()))
        print('reg:{reg}'.format(reg=reg_loss.item()))

        print_loss[-1].append(sft_loss.item())
        self.accelerator.backward(total_loss)

    def prepare_hfa_dataloader(self, train_file_path=args.train_file_path):
        self.logger.info(f"Load training data from {os.path.join(train_file_path)}")
        self.accelerator.print(f"Load training data from {os.path.join(train_file_path)}")
        
        hfa_dataloader = self.data_manager.load_train_data_adaptad(
            data_file_path = train_file_path,
            data_collator = self.data_manager.train_data_collator_adapt
        )
        
        # wrap with accelerator
        hfa_dataloader = self.accelerator.prepare(
            hfa_dataloader
        )

        return hfa_dataloader

    def init_prepare_train(self, train_file_path=args.train_file_path):
        # get dataloader
        temp_data = open(args.train_file_path,encoding='utf-8')
        temp_data = json.load(temp_data)
        # record raw dataset length
        dataset_length = len(temp_data)

        # get the placeholder dataloader
        placeholder_dataloader = self.data_manager.load_train_data_adaptad(
            data_file_path = args.train_file_path,
            data_collator = self.data_manager.train_data_collator_adapt
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        
        # Scheduler and math around the number of training steps.
        if args.max_train_steps is None:
            num_update_steps_per_epoch_per_train_file = math.ceil(
                math.ceil(
                    dataset_length / args.per_device_train_batch_size
                ) / args.gradient_accumulation_steps
            )
            args.max_train_steps = num_update_steps_per_epoch_per_train_file * args.num_train_epochs
        

        model, optimizer, _ = self.accelerator.prepare(
            self.model, optimizer, placeholder_dataloader
        )
        self.model = None

        total_batch_size = args.per_device_train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps
        self.logger.info("***** Running training *****", main_process_only=True)
        self.logger.info(f"  Num examples = {dataset_length}", main_process_only=True)
        self.logger.info(f"  Num training stages = {args.training_stage_num}", main_process_only=True)
        self.logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}", main_process_only=True)
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}", main_process_only=True)
        self.logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}", main_process_only=True)
        
        return model, optimizer, dataset_length

    def train(self):
        model, optimizer, dataset_length = self.init_prepare_train(
            train_file_path=args.train_file_path
        )
        training_stage = args.training_stage_num
        if self.accelerator.is_main_process:
            writer = SummaryWriter(args.log_path)
        
            if args.do_validation:
                model_to_save = self.accelerator.unwrap_model(model)
                dev_res = self.infer(
                                model = model_to_save,
                                infer_file_path = args.validation_file_path
                            )
                prefixes, suffixes = [], []
                infer_f1 ,infer_em = infer_evaluate(dev_res)
                self.logger.info(f"Step 0 | Dev avg  f1 {infer_f1}")

                model_to_save = None
                
        self.accelerator.wait_for_everyone()
        # Train!
        progress_bar = tqdm(
            range(
                math.ceil(
                    math.ceil(
                        math.ceil(
                            dataset_length / args.per_device_train_batch_size
                        ) / self.accelerator.num_processes
                    ) / args.gradient_accumulation_steps
                ) * args.num_train_epochs
            ),
            disable=not self.accelerator.is_local_main_process
        )
        completed_steps = 0
        best_step = -1
        last_infer_score = float('-inf')
        for epoch in range(args.num_train_epochs):
            if self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch} starts")
                self.accelerator.print(f"\nEpoch {epoch} starts")
            train_file_path = args.train_file_path
            if_get_new_dataloader = False
            if epoch == 0:
                if_get_new_dataloader = True
            else:
                pass
                
            torch.cuda.empty_cache()
            if if_get_new_dataloader:
                hfa_dataloader = None
                hfa_dataloader = self.prepare_hfa_dataloader(train_file_path)

            print_loss = [[] for i in range(training_stage)]
            for step, batch in enumerate(hfa_dataloader):
                model.train()
                with self.accelerator.accumulate(model):
                    self.compute_loss(model, batch, print_loss)
                    optimizer.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    completed_steps += 1
                    progress_bar.update(1)
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        print(print_loss)
                        print_loss = [sum(l) for l in print_loss]
                        print_loss = [l / self.accelerator.gradient_accumulation_steps for l in print_loss]
                        total_loss = sum(print_loss)
                    
                        print_loss_info = "\nstage_{}_loss: {:.4f}".format(training_stage, total_loss)
                        
                        print_loss_info += "".join(
                            [" | rank_{}_loss: {:.4f}".format(n+1, print_loss[n]) for n in range(training_stage-1)]
                        )
                        print_loss_info += " | sft_loss: {:.4f}".format(print_loss[training_stage-1])
                        
                        for n in range(training_stage-1):
                            writer.add_scalar("stage_{}/rank_{}_loss".format(training_stage, n+1), print_loss[n], completed_steps)
                        writer.add_scalar("stage_{}/sft_loss".format(training_stage), print_loss[training_stage-1], completed_steps)
                        
                        self.logger.info(f"Step {completed_steps} | " + print_loss_info)
                        writer.add_scalar("stage_{}/loss".format(training_stage), total_loss, completed_steps) # record on tensorboard                      
                        
                        if args.do_validation and (completed_steps % args.checkpointing_step == 0 or (step == len(hfa_dataloader)-1)):
                            model_to_save = self.accelerator.unwrap_model(model)

                            dev_res = self.infer(
                                model = model_to_save,
                                infer_file_path = args.validation_file_path
                            )
                            prefixes, suffixes = [], []
                            infer_f1,infer_em = infer_evaluate(dev_res)

                            self.logger.info(f"Step {completed_steps} | Dev avg f1 {infer_f1}")
                            if infer_f1 > last_infer_score:
                                best_step = completed_steps
                                self.logger.info(f"Step {completed_steps} checkpoint with higher Dev avg reward (the best checkpoint so far)")
                                last_infer_score = infer_f1
                            self.save_checkpoint(model_to_save,self.data_manager.tokenizer,os.path.join(args.output_dir, 'step_{}.bin'.format(completed_steps)))

                            model_to_save = None

                    print_loss = [[] for i in range(training_stage)]
                    self.accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
            if self.accelerator.sync_gradients and self.accelerator.is_main_process:
                model_to_save = self.accelerator.unwrap_model(model)
                self.save_checkpoint(model_to_save,self.data_manager.tokenizer,os.path.join(args.output_dir, 'epoch_{}.bin'.format(epoch)))
                self.logger.info(f"Epoch {epoch} checkpoint has been saved.")
                model_to_save = None
        if args.do_validation and self.accelerator.is_main_process:
            print("The best checkpoint is step_{}".format(best_step))
            os.symlink('step_{}.bin'.format(best_step), os.path.join(args.output_dir, 'best_checkpoint.bin'))
        if self.accelerator.is_main_process:
            record_list_total = self.accelerator.gather(record_list) 
            with open(args.choice_cache,'w',encoding='utf-8') as f:
                json.dump(record_list_total,f,ensure_ascii=False,indent=2)
            writer.close()
        
        return self.accelerator.unwrap_model(model)
    
    def infer(self, model, infer_file_path=None):
        torch.cuda.empty_cache()
        model.eval()

        eval_data = open(infer_file_path,encoding='utf-8')
        eval_data = json.load(eval_data)

        infer_data = eval_data[:500]

        length = []
        for l in infer_data:
            lens = 0
            p = l['question']
            lens += (len(p.split(" ")))
            length.append(lens)
        
        indices = list(range(len(length)))
        back_indices = indices
        infer_data = [infer_data[index] for index in indices]

        infer_batch_size = args.per_device_eval_batch_size                                
        infer_bar = tqdm(range(len(infer_data)), desc= "Inference on {}".format(infer_file_path))
        for sample_index in range(0,len(infer_data),infer_batch_size):
            if len(infer_data)-sample_index < infer_batch_size:
                infer_batch_size = len(infer_data)-sample_index

            prefixes = [l['question'] for l in infer_data[sample_index:sample_index+infer_batch_size]]
            suffixes = self.data_manager.infer_generate(model.pretrained_model, prefixes)
            for l, s in zip(infer_data[sample_index:sample_index+infer_batch_size], suffixes):
                l['infer'] = s
            infer_bar.update(infer_batch_size)
        torch.cuda.empty_cache()
        infer_data = [infer_data[index] for index in back_indices]
        return infer_data
        

    def save_checkpoint(self, model, tokenizer, path):
        if path is not None and path != '':
            torch.save(model.pretrained_model.state_dict(), path) 
        else:
            self.logger.error('No save path!', main_process_only=True)



