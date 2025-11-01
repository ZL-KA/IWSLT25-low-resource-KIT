from transformers import SeamlessM4Tv2ForSpeechToText, SeamlessM4Tv2ForTextToText, AutoProcessor
import argparse
from utils import seed_everything
import utils
import ds_loader
from datasets import load_dataset, Audio

iwslt_lang_map = {'gle': 'gle', 
'mltmasri':'mlt', 'mltcv':'mlt', 'iwsltrealtestmltmasri':'mlt', 'iwsltrealtestmltcv':'mlt',
'iwslt22ta':'arb', 'iwsltapcengvalid':'arb', 'bigc':'swh', 'apc_valid_normalize':'arb', 'apc_test_normalize':'arb',
'iwsltrealtestbembaasrtest1':'swh', 'iwsltrealtestbembaasrtest2':'swh', 'iwsltrealtestbembasttest':'swh'
}


epsilon_cutoff=0.02
temperature=1.0

def lowercase_and_remove_punct(example, items):
    arabic_filter = re.compile(r'[OUM]+/*|\u061F|\?|\!|\.')
    english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\.|\,|\"|\:')
    for column in items:
        text = example[column]
        # Remove punctuation
        if filter=='english':
            text = re.subn(english_filter, '', text)[0]
        elif filter=='arabic':
            text = re.subn(arabic_filter, '', text)[0].lower()
        # text = ''.join(char for char in text if char not in string.punctuation)
        example[column] = text
    return example

if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "12"  
    parser = argparse.ArgumentParser()
    # Main args
    parser.add_argument('--whattask', type=str, required=True, help='asr, mt, e2est, cascadedst, or e2estmbr')
    parser.add_argument('--generate_what_split',nargs='+', required=True, default='valid test', help='list of splits from valid and test; use nosplit if need')
    parser.add_argument('--ckp',type=str)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--no_ds_and_read_from_the_csv', type=str, default=None)
    
    # generation
    # parser.add_argument('--normalize_asr_prediction', action='store_true')
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dombr', action='store_true')
    parser.add_argument('--what_tgt_lang', type=str, default=None)
    parser.add_argument('--what_src_lang', type=str, default=None)

    
    # Saving
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--saving_suffix', type=str, default='suffix')
    parser.add_argument('--save_as', type=str, default=None)

    
    args = parser.parse_args()
    print(args)

    seed_everything()

    # Load dataset    
    if args.no_ds_and_read_from_the_csv:
        from datasets import Dataset
        import pandas as pd
        aaa=pd.read_csv(args.no_ds_and_read_from_the_csv, sep='|')
        dataset = Dataset.from_pandas(aaa)
        # dataset=load_dataset('csv', data_files=args.no_ds_and_read_from_the_csv, sep='|')['train']
        if args.whattask in ['asr', 'e2est']:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    else:
        dataset = ds_loader.load_dataset_iwslt(args.dataset)
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    
    # Do inference
    for split in args.generate_what_split:
        # What task?
        if args.whattask in ['asr', 'e2est']:
            # import pdb;pdb.set_trace()
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained(args.ckp)
            model.to(args.device)
            model.eval()
            
            if args.whattask == 'e2est':
                target_column='translation'
                pred_column='pred_translation'
                tgt_lang='eng'
            else:
                target_column='transcript'
                pred_column='pred_transcript'
                # Select target language, same as processor in utils.py
                if args.dataset in utils.ara_ds_names:
                    tgt_lang='arb'
                elif args.dataset in utils.bem_ds_names:
                    tgt_lang='swh'
                elif args.what_tgt_lang:
                    tgt_lang=args.what_tgt_lang
                else:
                    import pdb
                    pdb.set_trace()

            def do_asr_e2est(example):
                audio_inputs = processor(audios=example['audio']['array'], return_tensors="pt", sampling_rate = example['audio']['sampling_rate'], padding=True).to(args.device)
                try:
                    if args.dombr:
                        num_return_sequences=50
                        epsilon_cutoff=0.02
                        temperature=1.0
                        logits = model.generate(**audio_inputs, tgt_lang=tgt_lang, do_sample=True, num_return_sequences=num_return_sequences, epsilon_cutoff=epsilon_cutoff, temperature=temperature).cpu()
                        translated_text = processor.batch_decode(logits, skip_special_tokens=True)
                        example[pred_column] = "<SS>".join(translated_text)
                    else:
                        logits = model.generate(**audio_inputs, tgt_lang=tgt_lang, num_beams=args.num_beams, no_repeat_ngram_size=args.no_repeat_ngram_size)[0].cpu().squeeze()
                        translated_text = processor.decode(logits, skip_special_tokens=True)
                        example[pred_column] = translated_text
                except:
                    print('abnormal file do not work', example['audio']['file'])
                    example[pred_column]=' '
                return example
            
            # Do prediction
            # import pdb;pdb.set_trace()
            if split=='nosplit':
                results=dataset.map(do_asr_e2est)
            else:
                results=dataset[split].map(do_asr_e2est)

        elif args.whattask=='mt' or args.whattask=='cascaded_mt':
            model = SeamlessM4Tv2ForTextToText.from_pretrained(args.ckp)
            model.to(args.device)
            model.eval()
            # Settings
            num_beams=5
            no_repeat_ngram_size=3

            target_column='translation'
            pred_column='pred_translation'
            src_column='transcript'
            if args.whattask=='cascaded_mt':
                src_column='pred_transcript'
            tgt_lang='eng'
            
            if args.dataset=='bigc':
                src_lang='bem'
            elif args.dataset!=None and ('apc' in args.dataset or 'aeb' in args.dataset):
                src_lang='arb'
            elif args.what_src_lang:
                src_lang=args.what_src_lang
            else:
                import pdb;pdb.set_trace()
            
            # Define function
            def do_mt(example):
                try:
                    if args.dombr:
                        text_inputs = processor(text = example[src_column], src_lang=src_lang, return_tensors="pt").to(args.device)
                        num_return_sequences=50
                        epsilon_cutoff=0.02
                        temperature=1.0
                        decoder_input_ids = model.generate(**text_inputs, tgt_lang=tgt_lang, do_sample=True, num_return_sequences=num_return_sequences, epsilon_cutoff=epsilon_cutoff, temperature=temperature).cpu()
                        translated_text_from_text = processor.batch_decode(decoder_input_ids, skip_special_tokens=True)
                        example[pred_column] = "<SS>".join(translated_text_from_text)
                    else:
                        text_inputs = processor(text = example[src_column], src_lang=src_lang, return_tensors="pt").to(args.device)
                        decoder_input_ids = model.generate(**text_inputs, tgt_lang=tgt_lang, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)[0].cpu().tolist()
                        translated_text_from_text = processor.decode(decoder_input_ids, skip_special_tokens=True)
                        example[pred_column] = translated_text_from_text
                except:
                    # import pdb;pdb.set_trace()
                    print('abnormal sample not working', example)
                    example[pred_column]=' '
                return example
            # Do prediction
            if split=='nosplit':
                results=dataset.map(do_mt)
            else:
                results=dataset[split].map(do_mt)

        # import pdb;pdb.set_trace()
        # Save results
        if args.save_as:
            if 'audio' in results.column_names:
                results = results.remove_columns(['audio'])
            results.to_csv(args.save_as, sep='|')
        elif args.save_results:
            if 'audio' in results.column_names:
                results = results.remove_columns(['audio'])
            if args.ckp:
                # path = os.path.sep.join(args.ftedmodel_path.split(os.path.sep)[:-1])+'/results.csv'
                if args.ckp=='facebook/seamless-m4t-v2-large':
                    path='/project/OML/zli/iwslt2025/results' + f'/{split}results_{args.saving_suffix}.csv'
                else:
                    path = args.ckp + f'/{split}results_{args.saving_suffix}.csv'
            else:
                if args.normalize_transcript:
                    path = f'/project/OML/zli/iwslt2025/results/seamlessm4t_infer_{args.language}_task{args.task}_normtranscript_{result_saving_prefix}.csv'
                else:
                    path = f'/project/OML/zli/iwslt2025/results/seamlessm4t_infer_{args.language}_task{args.task}_{result_saving_prefix}.csv'
            results.to_csv(path, sep='|')
            print(f'save to {path}')
        else:
            pass    

        