from transformers import SeamlessM4Tv2ForSpeechToText, SeamlessM4Tv2ForTextToText, AutoProcessor, EarlyStoppingCallback

import argparse
from utils import seed_everything, DataCollatorWithPaddingSEAMLESSM4T, prepare_dataset_seamlessm4t_asr_e2est, prepare_dataset_seamlessm4t_mt
import ds_loader
import torch



if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "12"
    parser = argparse.ArgumentParser()
    # Main args
    parser.add_argument('--whattask', type=str, required=True,choices=['asr', 'mt', 'e2est'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--training_config', type=str, required=True)
    
    # Train
    parser.add_argument('--use_rdrop', action='store_true')
    parser.add_argument('--rdrop_alpha', type=float,)
    parser.add_argument('--num_pass', type=int, default=3)
    parser.add_argument('--dropout_value', type=float, default=0.3)
    parser.add_argument('--use_peft', action='store_true')
    parser.add_argument('--resume_ckp', default=None)
    
    # Saving
    parser.add_argument('--save_path', type=str, required=True)
    
    
    args = parser.parse_args()
    print(args)

    seed_everything()
    
    
    from omegaconf import OmegaConf
    training_config = OmegaConf.load(args.training_config)
    print('training_config: ', training_config)
    
    # Load dataset
    
    dataset = ds_loader.load_dataset_iwslt(args.dataset)

    

    if args.resume_ckp:
        model_ckp = args.resume_ckp
    else:
        model_ckp = "facebook/seamless-m4t-v2-large"
    print(f'using model ckp: {model_ckp}')

    
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large") # Same processor
    if args.whattask in ['asr', 'e2est']:
        # Load model
        if args.use_rdrop:
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_ckp,
                                    activation_dropout=args.dropout_value,
                                    adaptor_dropout=args.dropout_value,
                                    attention_dropout=args.dropout_value,
                                    dropout=args.dropout_value,
                                    speech_encoder_dropout=args.dropout_value,
                                    var_pred_dropout=args.dropout_value)
        elif args.use_peft:
            import pdb;pdb.set_trace()
            from peft import get_peft_model, LoraConfig, TaskType
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_ckp, device_map='auto')
            for param in model.parameters():
                param.requires_grad = False  # freeze the model - train adapters later
                if param.ndim == 1:
                    # cast the small parameters (e.g. layernorm) to fp32 for stability
                    param.data = param.data.to(torch.float32)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            TARGET_MODULES=["q_proj", "v_proj", "linear_q", "linear_v"]
            LORA_ALPHA=32
            LORA_R=8
            LORA_DROPOUT=0.1
            BIAS='lora_only'
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, target_modules=TARGET_MODULES, inference_mode=False, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias=BIAS)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_ckp)
        
        model.to('cuda')
        # Process dataset
        train_dataset = dataset['train'].map(prepare_dataset_seamlessm4t_asr_e2est, fn_kwargs={'model': model, 'processor':processor, 'task':args.whattask, 'dataset':args.dataset})
        dev_dataset = dataset['valid'].map(prepare_dataset_seamlessm4t_asr_e2est, fn_kwargs={'model': model, 'processor':processor, 'task':args.whattask, 'dataset':args.dataset},remove_columns=['audio'])
        data_collator = DataCollatorWithPaddingSEAMLESSM4T(processor=processor, padding=True, model=model)
    elif args.whattask=='mt':
    
        # Load model
        if args.use_rdrop:
            model = SeamlessM4Tv2ForTextToText.from_pretrained(model_ckp,
                                    activation_dropout=args.dropout_value,
                                    adaptor_dropout=args.dropout_value,
                                    attention_dropout=args.dropout_value,
                                    dropout=args.dropout_value,
                                    speech_encoder_dropout=args.dropout_value,
                                    var_pred_dropout=args.dropout_value)
        elif args.use_peft:
            import pdb;pdb.set_trace()
        else:
            model = SeamlessM4Tv2ForTextToText.from_pretrained(model_ckp)
    
        model.to('cuda')
        # Process dataset
        columns_to_remove = dataset['train'].column_names
        max_length=128
        padding='longest'
        train_dataset = dataset['train'].map(prepare_dataset_seamlessm4t_mt, fn_kwargs={'the_dataset': args.dataset, 'processor':processor, 'padding':padding, 'max_length': max_length}, remove_columns=columns_to_remove)
        dev_dataset = dataset['valid'].map(prepare_dataset_seamlessm4t_mt, fn_kwargs={'the_dataset': args.dataset, 'processor':processor, 'padding':padding, 'max_length': max_length}, remove_columns=columns_to_remove)
        from utils import DataCollatorForSeq2SeqSEAMLESSM4TMT
        data_collator = DataCollatorForSeq2SeqSEAMLESSM4TMT(tokenizer=processor.tokenizer, model=model, label_pad_token_id=processor.tokenizer.pad_token_id, padding=padding, max_length=max_length, return_tensors='pt')
        
        
    else:
        raise NotImplementedError
    # Do training
    output_path = args.save_path
    print(model)
    

    if args.whattask=='mt':
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
        training_args = Seq2SeqTrainingArguments(
                output_dir=output_path,
                group_by_length=training_config.trainer.group_by_length,
                per_device_train_batch_size=training_config.trainer.per_device_train_batch_size,
                per_device_eval_batch_size=training_config.trainer.per_device_eval_batch_size,
                gradient_accumulation_steps=training_config.trainer.gradient_accumulation_steps,
                evaluation_strategy=training_config.trainer.evaluation_strategy,
                num_train_epochs=training_config.trainer.num_train_epochs,
                fp16=training_config.trainer.fp16,
                gradient_checkpointing=training_config.trainer.gradient_checkpointing,
                save_steps=training_config.trainer.save_steps,
                eval_steps=training_config.trainer.eval_steps,
                logging_steps=training_config.trainer.logging_steps,
                learning_rate=training_config.trainer.learning_rate,
                weight_decay=training_config.trainer.weight_decay,
                warmup_steps=training_config.trainer.warmup_steps,
                save_total_limit=training_config.trainer.save_total_limit,
                metric_for_best_model=training_config.trainer.metric_for_best_model,
                greater_is_better=training_config.trainer.greater_is_better,
                seed=training_config.trainer.seed,
                data_seed=training_config.trainer.data_seed,
                label_smoothing_factor=training_config.trainer.label_smoothing_factor,
                lr_scheduler_type=training_config.trainer.lr_scheduler_type,
                load_best_model_at_end=training_config.trainer.load_best_model_at_end,
            )

        # RDROP settings
        if args.use_rdrop:
            training_args.do_rdrop=True
            training_args.num_pass=args.num_pass
            training_args.kl_div_alpha=args.rdrop_alpha
            training_args.loss_type='x_loss'
        else:
            training_args.do_rdrop=False


    
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=processor.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=training_config.trainer.early_stopping_patience)])
            

        
    elif args.whattask in ['asr', 'e2est']:
        from transformers import TrainingArguments, Trainer
        training_args = TrainingArguments(
                output_dir=output_path,
                group_by_length=training_config.trainer.group_by_length,
                per_device_train_batch_size=training_config.trainer.per_device_train_batch_size,
                per_device_eval_batch_size=training_config.trainer.per_device_eval_batch_size,
                gradient_accumulation_steps=training_config.trainer.gradient_accumulation_steps,
                evaluation_strategy=training_config.trainer.evaluation_strategy,
                num_train_epochs=training_config.trainer.num_train_epochs,
                fp16=training_config.trainer.fp16,
                gradient_checkpointing=training_config.trainer.gradient_checkpointing,
                save_steps=training_config.trainer.save_steps,
                eval_steps=training_config.trainer.eval_steps,
                logging_steps=training_config.trainer.logging_steps,
                learning_rate=training_config.trainer.learning_rate,
                weight_decay=training_config.trainer.weight_decay,
                warmup_steps=training_config.trainer.warmup_steps,
                save_total_limit=training_config.trainer.save_total_limit,
                metric_for_best_model=training_config.trainer.metric_for_best_model,  # 'eval_loss',
                # False if metric_for_best_model is not set, or set to "loss" or "eval_loss".
                greater_is_better=training_config.trainer.greater_is_better,
                seed=training_config.trainer.seed,
                data_seed=training_config.trainer.data_seed,
                label_smoothing_factor=training_config.trainer.label_smoothing_factor,
                lr_scheduler_type=training_config.trainer.lr_scheduler_type,
                load_best_model_at_end=training_config.trainer.load_best_model_at_end
            )
        
        # RDROP settings
        if args.use_rdrop:
            training_args.do_rdrop=True
            training_args.num_pass=args.num_pass
            training_args.kl_div_alpha=args.rdrop_alpha
            training_args.loss_type='x_loss'
        else:
            training_args.do_rdrop=False

        
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=processor.feature_extractor,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=training_config.trainer.early_stopping_patience)])
    

    # import pdb;pdb.set_trace()
    trainer.train()
    
    