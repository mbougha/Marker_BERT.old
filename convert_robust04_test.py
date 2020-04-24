import argparse
from Processors.robust04 import convert_eval_dataset

def main():
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_folder", default=None, type=str, required=True)
    parser.add_argument("--queries_path", default=None, type=str, required=True,
                            help="the path to the test queries .tsv file.")
    parser.add_argument("--run_path", default=None, type=str, required=True,
                            help="the path to the run file .tsv file : q_id, doc_id, rank, pred.")
    parser.add_argument("--collection_path", default=None, type=str, required=True,
                            help="the path to the documents .tsv file: doc_id, url, title, doc_body.")
    parser.add_argument("--num_eval_docs", default=1000, type=int, required=False,
                            help="the number of documents retrieved per query.")
    parser.add_argument("--sentence_level", default=False, type=bool, required=False,
                            help="create a sentence split output.")
    parser.add_argument("--set_name", default=None, type=str, required=True,
                            help="set name.")
    args = parser.parse_args()
    
    convert_eval_dataset(args.output_folder, args.queries_path, args.run_path, args.collection_path, args.set_name, args.num_eval_docs, args.sentence_level)
              

if __name__ == "__main__":
    main()