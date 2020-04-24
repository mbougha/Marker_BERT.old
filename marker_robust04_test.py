import argparse
from Processors import Robust04Processor, get_marker, DocumentHandle, DocumentSplitterHandle
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AlbertTokenizer


MODEL_CLASSES = {
    "bert": BertTokenizer,
    "roberta":  RobertaTokenizer,
    "distilbert": DistilBertTokenizer,
    "albert" : AlbertTokenizer,
}
HANDLE = {
    "sentence" : DocumentHandle,
    "split" : DocumentSplitterHandle,
}

def main():
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--strategy", default=None, type=str, required=True,
                            help="the marking strategy in ('base', 'un_mark_pass', 'un_mark_pair', 'mu_mark_pass', 'mu_mark_pair')")
    parser.add_argument("--data_path", default=None, type=str, required=True,
                            help="The input data file.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output dir, data to be saved.")
    parser.add_argument("--tokenizer_model", default='bert', type=str, required=False,
                            help=f"The name of the model must be in : {set(MODEL_CLASSES.keys())}")
    parser.add_argument("--tokenizer_init", default='bert-base-uncased', type=str, required=False,
                            help=f"path to the initialization tokenizer or name in transformers")
    parser.add_argument("--handle", default='sentence', type=str, required=False,
                            help="handle the document length : 'sentence' or 'split' ==> fixed size chunks ")
    parser.add_argument("--max_seq_len", default=512, type=int, required=False,
                            help=" max seq length for the transformer.")
    parser.add_argument("--max_query_len", default=64, type=int, required=False,
                            help=" max query length in the input sequence of the transformer.")
    parser.add_argument("--max_title_len", default=64, type=int, required=False,
                            help=" max title length in the input sequence of the transformer.")
    parser.add_argument("--chunk_size", default=384, type=int, required=False,
                            help=" split the sequence into fixed size chunks of size chunk_size.")
    parser.add_argument("--stride", default=192, type=int, required=False,
                            help="if split into overlapping chunks set this stride.")
    parser.add_argument("--set_name", default=None, type=str, required=True,
                            help="set name.")

    args = parser.parse_args()

    tokenizer_class = MODEL_CLASSES[args.tokenizer_model.lower()]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_init)

    marker = get_marker(args.strategy.lower())

    handle_class = HANDLE[args.handle.lower()]
    handle = handle_class(tokenizer, args.max_seq_len, args.max_query_len, args.max_title_len, args.chunk_size, args.stride)
    doc_processor = Robust04Processor(handle,marker)
            
    doc_processor.prepare_inference_dataset(args.data_path, args.output_dir,args.set_name)

if __name__ == "__main__":
    main()