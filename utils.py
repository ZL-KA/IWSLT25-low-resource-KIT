
import json
import random
import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

bem_ds_names = ['bigc', 'bigc_and_tts_yining_30k', 'bigc_and_tts_yining_60k', 'bigc_and_tts_yining_120k', 'bigc_and_tts_nam_30k', 'bigc_and_tts_nam_60k', 'bigc_and_tts_nam_120k', 'bembamtall']
ara_ds_names = ['apc_combined_normalize', 'apc_combined_and_ufal_and_flores_normalize', 'apc_combined_and_ufal_normalize',
        'apc_validonly_and_ufal_and_flores_normalize', 'apc_validonly_and_ufal_normalize', 'apc_validonly_normalize',
        'apc_valid_normalize', 'apc_test_normalize',
        'aeb_normalize',
        'apc_mt_ufal_flores_normalize', 'apc_mt_ufal_normalize',
        'apc_tts_normalize_30k', 'apc_tts_normalize_15k', 'apc_tts_normalize_60k',
        'apc_tts_normalize_30k_withvalid', 'apc_tts_normalize_15k_withvalid', 'apc_tts_normalize_60k_withvalid',
        'apc_sysst_ldc2005s08', 'apc_sysst_ldc2005s08_half', 'apc_sysst_ldc2005s08_withvalid', 'apc_sysst_ldc2005s08_half_withvalid',
        'new_apc_sysst_ldc2005s08', 'new_apc_sysst_ldc2005s08_half', 'new_apc_sysst_ldc2005s08_withvalid', 'new_apc_sysst_ldc2005s08_half_withvalid', 
        'new_apc_sysst_ldc2005s08_onequarter', 'new_apc_sysst_ldc2005s08_oneeighth']

    
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText, SeamlessM4TProcessor, SeamlessM4Tv2ForTextToText
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2SpeechEncoder, SeamlessM4Tv2Decoder


def load_model_processor(args):

    if args.pretrained_seamlessm4t_model:
        print('*********** Load pretrained model')
        hfmodel = args.pretrained_seamlessm4t_model
    else:
        print('*********** Load from Huggingface')
        hfmodel = "facebook/seamless-m4t-v2-large"
    hfmodel = "facebook/seamless-m4t-v2-large"
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    if args.whattask == 'mt':
        model = SeamlessM4Tv2ForTextToText.from_pretrained(hfmodel)
    elif args.whattask == 'st' or args.whattask == 'asr':
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(hfmodel)
    else:
        import pdb
        pdb.set_trace()
    return model, processor

def prepare_dataset_seamlessm4t_asr_e2est(example, model, processor, task, dataset):
    if task=='e2est':
        example['input_features']=processor(audios=example['audio']['array'], return_tensors="pt", sampling_rate = example['audio']['sampling_rate']).input_features[0]
        example["labels"] = processor.tokenizer(text_target=example["translation"], tgt_lang='eng').input_ids
    elif task=='asr':
        example['input_features']=processor(audios=example['audio']['array'], return_tensors="pt", sampling_rate = example['audio']['sampling_rate']).input_features[0]
        if dataset in ara_ds_names:
            tgt_lang='arb'
        elif dataset in bem_ds_names:
            tgt_lang='swh'
        else:
            import pdb
            pdb.set_trace()
        example["labels"] = processor.tokenizer(text_target=example["transcript"], tgt_lang=tgt_lang).input_ids
    else:
        import pdb
        pdb.set_trace()
        print('Wrong task')
    return example



def prepare_dataset_seamlessm4t_mt(example, the_dataset, processor, padding, max_length):
    # import pdb; pdb.set_trace()
    if the_dataset in bem_ds_names:
        src_lang='bem'
        src_column='transcript'
        tgt_column='translation'
    elif the_dataset in ara_ds_names:
        src_lang='arb'
        src_column='transcript'
        tgt_column='translation'
    else:
        import pdb; pdb.set_trace()

    # use supported lang_tag
    if src_lang=='bem':
        src_lang='swh'
    elif src_lang=='arb':
        src_lang=='arb'
    else:
        import pdb; pdb.set_trace()
    
    model_inputs = processor.tokenizer(text=example[src_column], src_lang=src_lang, padding=padding, max_length=max_length, truncation=True)
    labels = processor.tokenizer(text_target=example[tgt_column], tgt_lang='eng', padding=padding, max_length=max_length, truncation=True) # Target_lang is always eng

    if padding=='longest':
        labels["input_ids"] = [(l if l != processor.tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
    model_inputs['labels']=labels['input_ids']
    # import pdb;pdb.set_trace()
    return model_inputs




from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import shift_tokens_right
from typing import Any


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
@dataclass
class DataCollatorForSeq2SeqSEAMLESSM4TMT:
    
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        assert isinstance(labels, list)
        tensor_data = [torch.tensor(seq) for seq in labels]
        padded_labels=pad_sequence(tensor_data, batch_first=True, padding_value= self.model.config.pad_token_id)

        decoder_input_ids = shift_tokens_right(padded_labels, self.model.config.pad_token_id, self.model.config.decoder_start_token_id)
        features['decoder_input_ids'] = decoder_input_ids

        return features




@dataclass
class DataCollatorWithPaddingSEAMLESSM4T:
    model: Any
    processor: SeamlessM4TProcessor
    padding: Union[bool, str] = True
    

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding,return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in the previous tkenization step, cut bos token here as it is append later anyways.
        if (labels[:,0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:,1:]

        batch["labels"] = labels

        # To use label_smoothing, add "decoder_input_ids"

        decoder_input_ids = shift_tokens_right(labels, self.model.config.pad_token_id, self.model.config.decoder_start_token_id)
        batch['decoder_input_ids'] = decoder_input_ids
        
        
        return batch


