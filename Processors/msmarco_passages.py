import logging
import collections
import tensorflow as tf
import os, time

from .processor_utils import DataProcessor, strip_html_xml_tags, clean_text
from .marker_utils import get_marker

logger = logging.getLogger(__name__)


# util functions for msmarco passage dataset
def convert_eval_dataset(output_folder, 
                        qrels_path, 
                        queries_path, 
                        run_path, 
                        collection_path, 
                        set_name,
                        num_eval_docs):
    
    if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    qrels = _load_qrels(qrels_path, set_name)

    queries = _load_queries(path=queries_path)
    run = _load_run(path=run_path)
    data = _merge(qrels=qrels, run=run, queries=queries)

    print('Loading Collection...')
    collection = _load_collection(collection_path)

    print('Converting to TFRecord...')
    _convert_dataset(data,collection, set_name, num_eval_docs, output_folder)

    print('Done!')

def _convert_dataset(data, 
                        collection, 
                        set_name, 
                        num_eval_docs, 
                        output_folder):

    output_path = output_folder + f'/run_{set_name}_full.tsv'
    start_time = time.time()
    random_title = list(collection.keys())[0]

    with open(output_path, 'w') as writer:
        for i, query_id in enumerate(data):
                query, qrels, doc_titles = data[query_id]

                clean_query = clean_text(query)

                doc_titles = doc_titles[:num_eval_docs]

                # Add fake docs so we always have max_docs per query.
                doc_titles += max(0, num_eval_docs - len(doc_titles)) * [random_title]

                labels = [
                    1 if doc_title in qrels else 0 
                    for doc_title in doc_titles
                ]

                len_gt_query = len(qrels)

                for label, doc_title in zip(labels, doc_titles):
                    _doc = strip_html_xml_tags(collection[doc_title])
                    clean_doc = clean_text(_doc)

                    writer.write("\t".join((query_id, doc_title, clean_query, clean_doc, str(label), str(len_gt_query))) + "\n")

                if i % 1000 == 0:
                    print('wrote {} of {} queries'.format(i, len(data)))
                    time_passed = time.time() - start_time
                    est_hours = (len(data) - i) * time_passed / (max(1.0, i) * 3600)
                    print('estimated total hours to save: {}'.format(est_hours))


def _load_qrels(path, set_name):
    """Loads qrels into a dict of key: query_id, value: list of relevant doc ids."""
    qrels = collections.defaultdict(set)

    relevance_threshold = 2 if set_name=='test' else 1

    with open(path) as f:
        for i, line in enumerate(f):
                query_id, _, doc_id, relevance = line.rstrip().split('\t')
                if int(relevance) >= relevance_threshold:
                    qrels[query_id].add(doc_id)
                if i % 1000 == 0:
                    print('Loading qrels {}'.format(i))
    return qrels


def _load_queries(path):
    """Loads queries into a dict of key: query_id, value: query text."""
    queries = {}
    with open(path) as f:
        for i, line in enumerate(f):
                query_id, query = line.rstrip().split('\t')
                queries[query_id] = query
                if i % 1000 == 0:
                    print('Loading queries {}'.format(i))
    return queries


def _load_run(path):
    """Loads run into a dict of key: query_id, value: list of candidate doc ids."""
    # We want to preserve the order of runs so we can pair the run file with the
    # TFRecord file.
    run = collections.OrderedDict()
    with open(path) as f:
        for i, line in enumerate(f):
                query_id, doc_title, rank = line.split('\t')
                if query_id not in run:
                    run[query_id] = []
                run[query_id].append((doc_title, int(rank)))
                if i % 1000000 == 0:
                    print('Loading run {}'.format(i))
    # Sort candidate docs by rank.
    sorted_run = collections.OrderedDict()
    for query_id, doc_titles_ranks in run.items():
            sorted(doc_titles_ranks, key=lambda x: x[1])
            doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
            sorted_run[query_id] = doc_titles

    return sorted_run


def _merge(qrels, run, queries):
    """Merge qrels and runs into a single dict of key: query, 
        value: tuple(relevant_doc_ids, candidate_doc_ids)"""
    data = collections.OrderedDict()
    for query_id, candidate_doc_ids in run.items():
            query = queries[query_id]
            relevant_doc_ids = set()
            if qrels:
                relevant_doc_ids = qrels[query_id]
            data[query_id] = (query, relevant_doc_ids, candidate_doc_ids)
    return data


def _load_collection(path):
    """Loads tsv collection into a dict of key: doc id, value: doc text."""
    collection = {}
    with open(path) as f:
        for i, line in enumerate(f):
                doc_id, doc_text = line.rstrip().split('\t')
                collection[doc_id] = doc_text.replace('\n', ' ')
                if i % 1000000 == 0:
                    print('Loading collection, doc {}'.format(i))
    return collection


def convert_train_dataset(train_dataset_path,
                        output_folder,
                        tokenizer,
                            ):

    print('Converting to Train to pairs tsv...')

    start_time = time.time()

    print('Counting number of examples...')
    num_lines = sum(1 for line in open(train_dataset_path, 'r'))
    print('{} examples found.'.format(num_lines))
    
    with open(f'{output_folder}/train_pairs.tsv', 'w') as writer:
        with open(train_dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    time_passed = int(time.time() - start_time)
                    print('Processed training set, line {} of {} in {} sec'.format(
                        i, num_lines, time_passed))
                    hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
                    print('Estimated hours remaining to write the training set: {}'.format(
                        hours_remaining))

                query, positive_doc, negative_doc = line.rstrip().split('\t')

                clean_query = clean_text(query)
                positive_doc = strip_html_xml_tags(positive_doc)
                positive_doc = clean_text(positive_doc)
                negative_doc = strip_html_xml_tags(negative_doc)
                negative_doc = clean_text(negative_doc)

                out.write('\t'.join([clean_query, positive_doc, 1])+'\n')
                out.write('\t'.join([clean_query, negative_doc, 0])+'\n')     

    print("writer closed, DONE !")
    print(f'writer closed with {i*2} lines')

class MsMarcoPassageProcessor(DataProcessor):

    def __init__(self, 
                passage_handle,
                marker
                ):
        super().__init__()
        self.passage_handle = passage_handle
        self.marker = marker

    def get_train_dataset (self, data_path, batch_size):
        return self.passage_handle.get_train_dataset(data_path, batch_size)
    
    def get_eval_dataset (self, data_path, batch_size, num_skip=0):
        return self.passage_handle.get_eval_dataset(data_path, batch_size, num_skip)

    def prepare_train_dataset( self,
                             data_path, 
                             output_dir,
                             ):
        tf_writer = tf.io.TFRecordWriter(f"{output_dir}/dataset_train.tf")
        tsv_writer = open(f"{output_dir}/pairs_train.tsv", 'w')

        start_time = time.time()

        print('Counting number of examples...')
        num_lines = sum(1 for line in open(data_path, 'r'))
        print('{} examples found.'.format(num_lines))

        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    time_passed = int(time.time() - start_time)
                    print('Processed training set, line {} of {} in {} sec'.format(
                        i, num_lines, time_passed))
                    hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
                    print('Estimated hours remaining to write the training set: {}'.format(
                        hours_remaining))

                query, doc, label = line.rstrip().split('\t')
                q, p = self.marker.mark(query, doc)
                # write tfrecord
                self.passage_handle.write_train_example(tf_writer, q, [p], [int(label)])
                tsv_writer.write(f"{q}\t{p}\t{label}\n")
        tf_writer.close()
        tsv_writer.close()

    def prepare_inference_dataset( self,
                             data_path, 
                             output_dir,
                             set_name, ):
        tf_writer = tf.io.TFRecordWriter(f"{output_dir}/dataset_{set_name}.tf")
        tsv_writer = open(f"{output_dir}/pairs_{set_name}.tsv", 'w')

        start_time = time.time()

        print('Counting number of examples...')
        num_lines = sum(1 for line in open(data_path, 'r'))
        print('{} examples found.'.format(num_lines))

        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    time_passed = int(time.time() - start_time)
                    print('Processed training set, line {} of {} in {} sec'.format(
                        i, num_lines, time_passed))
                    hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
                    print('Estimated hours remaining to write the training set: {}'.format(
                        hours_remaining))
                
                qid, pid, query, doc, label, len_gt_query = line.rstrip().split('\t')
                q, p = self.marker.mark(query, doc)

                # write tfrecord
                self.passage_handle.write_eval_example(tf_writer, q, [p], [int(label)], qid, [pid], int(len_gt_query))
                tsv_writer.write(f"{qid}\t{q}\t{pid}\t{p}\t{label}\t{len_gt_query}\n")
        tf_writer.close()
        tsv_writer.close()



