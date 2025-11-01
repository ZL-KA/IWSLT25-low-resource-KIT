import sys
import re
do_additional_aeb_normlizer=False
do_additional_apc_normlizer=False
sys.path.append("/project/i13t81/eugan/ASR/IWSLT25/data/")
from normalize import normalize_line

arabic_filter = re.compile(r'[OUM]+/*|\u061F|\?|\!|\.')
english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\.|\,|\"|\:')

def lowercase_and_remove_punct(example, items, filter):
    for column in items:
        try:
            text = example[column].lower()
            text = re.subn(norm_filter, '', text)[0]
            if do_additional_aeb_normlizer:
                text = normalize_line(text, 'tunisian')
            if do_additional_aeb_normlizer:
                text = normalize_line(text, 'levantine')
            example[column] = text
        except:
            print(f'error in processing example with item {column} with value:', example[column])
            example[column] = ' '
    return example






results_path=sys.argv[1]
ref_conlumn_name=sys.argv[2]
pred_column_name=sys.argv[3]
lang_id = sys.argv[4]

assert lang_id in ['bem', 'apc', 'aeb']



if pred_column_name=='pred_transcript':
    evaluate_asr=True
    evaluate_mt=False
    if lang_id=='bem':
        norm_filter=english_filter
    else:
        norm_filter=arabic_filter
        if lang_id == 'apc':
            do_additional_aeb_normlizer=True # Since the ASR and trained with the new normalizer
        elif lang_id=='aeb':
            do_additional_apc_normlizer=True
        else:
            print('something wrong with results_path')
            import pdb;pdb.set_trace()
elif pred_column_name=='pred_translation' or pred_column_name=='llamape_prediction':
    evaluate_asr=False
    evaluate_mt=True
    norm_filter=english_filter


from datasets import load_dataset, Dataset
ds=load_dataset('csv', data_files=results_path, delimiter='|')['train']



print(ds)
# For bigc specifically
if ref_conlumn_name=='translation':
    ds=ds.filter(lambda example: example["translation"] is not None)
    print('after removing samples with None reference')
    print(ds)





# Do normalization
print('normalize columns: ', ref_conlumn_name, pred_column_name)
ds = ds.map(lowercase_and_remove_punct, fn_kwargs={'items':[ref_conlumn_name, pred_column_name], 'filter':norm_filter})


if evaluate_mt:
    import sacrebleu
    import evaluate
    
    
    def calculate_metrics(references, predictions, sources):
        # BLEU
        bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
        # chrF
        chrf_score = sacrebleu.corpus_chrf(predictions, [references]).score
        # TER
        ter_score = sacrebleu.corpus_ter(predictions, [references]).score
        # import pdb;pdb.set_trace()
        print('bleu_score: ', round(bleu_score,2))
        print('chrf_score: ', round(chrf_score,2)) 
        print('ter_score: ', round(ter_score,2))
        # if lang_id!='bem':
        #     comet_metric=evaluate.load('comet', 'wmt20-comet-da')
        #     comet_score = comet_metric.compute(predictions=predictions, references=references, sources=sources)
        #     print('comet: ', [round(v, 3) for v in comet_score["scores"]])

    
    calculate_metrics(references=ds[ref_conlumn_name], predictions=ds[pred_column_name], sources=ds['transcript'])

if evaluate_asr:
    from evaluate import load
    wer=load("wer")
    cer=load('cer')
    cer_result=round(cer.compute(references=ds[ref_conlumn_name], predictions=ds[pred_column_name])*100, 2)
    wer_result=round(wer.compute(references=ds[ref_conlumn_name], predictions=ds[pred_column_name])*100, 2)
    print(f'Overall CER: {cer_result}, WER: {wer_result}')

print(ds[0])
print('results_path: ', results_path)