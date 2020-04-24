from Processors import MsMarcoDocumentProcessor, get_marker, DocumentHandle, DocumentSplitterHandle
from transformers import BertTokenizer


marker = get_marker('un_mark_pair')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
handle = DocumentHandle(tokenizer, 512)
doc_processor = MsMarcoDocumentProcessor(handle,marker)

eval_batch_size = 128
filename = "/projets/iris/PROJETS/lboualil/workdata/msmarco-passage/MarkedBERT_large/My_MsMarco_TFRecord/trec/docs/sentence/un_mark_pair/dataset_test.tf"
eval_dataset, num_eval_examples = doc_processor.get_eval_dataset(filename, eval_batch_size)

print(num_eval_examples)