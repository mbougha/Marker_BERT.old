import argparse
from Processors import MsMarcoPassageProcessor, get_marker, PassageHandle
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AlbertTokenizer


MODEL_CLASSES = {
    "bert": BertTokenizer,
    "roberta":  RobertaTokenizer,
    "distilbert": DistilBertTokenizer,
    "albert" : AlbertTokenizer,
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
    parser.add_argument("--max_seq_len", default=512, type=int, required=False,
                            help=" max seq length for the transformer.")
    parser.add_argument("--max_query_len", default=64, type=int, required=False,
                            help=" max query length in the input sequence of the transformer.")
    parser.add_argument("--set_name", default='test', type=str, required=False,
                            help="set name.")

    args = parser.parse_args()

    tokenizer_class = MODEL_CLASSES[args.tokenizer_model.lower()]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_init)

    marker = get_marker(args.strategy.lower())

    handle = PassageHandle(tokenizer, args.max_seq_len, args.max_query_len)
    pass_processor = MsMarcoPassageProcessor(handle,marker)
            
    pass_processor.prepare_inference_dataset(args.data_path, args.output_dir,args.set_name)

if __name__ == "__main__":
    main()