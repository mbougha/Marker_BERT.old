import tensorflow as tf
from tensorflow.python.client import device_lib

QUERY_MAX_LEN = 64

def _extract_fn_train(data_record, max_seq_len):
    features = {
          "query_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "doc_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "label": tf.io.FixedLenFeature([], tf.int64),
      }
    sample = tf.io.parse_single_example(data_record, features)
      
    query_ids = tf.cast(sample["query_ids"], tf.int32) # max length with special tokens
    doc_ids = tf.cast(sample["doc_ids"], tf.int32) #max length with special tokens
    label_ids = tf.cast(sample["label"], tf.int32)
    
    query_ids_without_sep = query_ids[:-1]
    query_ids_trunc = tf.concat((query_ids_without_sep[:QUERY_MAX_LEN-1], query_ids[-1:]),0) # add SEP end
    doc_ids_without_markers = doc_ids[1:-1] # remove CLS and SEP
    
    input_ids = tf.concat((query_ids_trunc, doc_ids_without_markers), 0) # need [SEP] at last and cut at 512
    input_ids = tf.concat((input_ids[:511],doc_ids[-1:]), 0)

    input_mask = tf.ones_like(input_ids)

    query_segment_id = tf.zeros_like(query_ids_trunc)
    doc_segment_id = tf.ones_like(doc_ids[1:])
    segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)
    segment_ids = segment_ids[:512]

    features = {
          "input_ids": input_ids,
          "attention_mask": input_mask,
          "token_type_ids": segment_ids,
      }
      
    return (features, label_ids)
  
def _extract_fn_eval(data_record, max_seq_len):
    features = {
          "q_id" : tf.io.FixedLenFeature([], tf.string),
          "d_id" : tf.io.FixedLenFeature([], tf.string),
          "query_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "doc_ids": tf.io.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "label": tf.io.FixedLenFeature([], tf.int64),
          "len_gt_titles": tf.io.FixedLenFeature([], tf.int64),
      }
    sample = tf.io.parse_single_example(data_record, features)
    
    q_id = tf.strings.to_number(
            sample['q_id'], out_type=tf.dtypes.int32
    )
    d_id = tf.strings.to_number(
            sample['d_id'], out_type=tf.dtypes.int32
    )
    query_ids = tf.cast(sample["query_ids"], tf.int32) # max length with special tokens
    doc_ids = tf.cast(sample["doc_ids"], tf.int32) #max length with special tokens
    label_ids = tf.cast(sample["label"], tf.int32)
    len_gt_titles = tf.cast(sample["len_gt_titles"], tf.int32)
    
    query_ids_without_sep = query_ids[:-1]
    query_ids_trunc = tf.concat((query_ids_without_sep[:QUERY_MAX_LEN-1], query_ids[-1:]),0) # add SEP end
    doc_ids_without_markers = doc_ids[1:-1] # remove CLS and SEP
    
    input_ids = tf.concat((query_ids_trunc, doc_ids_without_markers), 0) # need [SEP] at last and cut at 512
    input_ids = tf.concat((input_ids[:511],doc_ids[-1:]), 0)

    input_mask = tf.ones_like(input_ids)

    query_segment_id = tf.zeros_like(query_ids_trunc)
    doc_segment_id = tf.ones_like(doc_ids[1:])
    segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)
    segment_ids = segment_ids[:512]

    features = {
          "q_id" : q_id,
          "d_id" : d_id,
          "input_ids": input_ids,
          "attention_mask": input_mask,
          "token_type_ids": segment_ids,
          "len_gt_titles": len_gt_titles,
      }
      
    return (features, label_ids)

def _get_dataset_train(dataset, batch_size, seq_length, num_skip=0):
    dataset = dataset.map( lambda record : _extract_fn_train(record, seq_length)).prefetch(batch_size*1000)
    #count = dataset.reduce(0, lambda x, _: x + 1)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=1000, seed=42)
    dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=({
                    "input_ids": [seq_length],
                    "attention_mask": [seq_length],
                    "token_type_ids": [seq_length]},
                    []
                ),
                padding_values=({
                    "input_ids": 0,
                    "attention_mask": 0,
                    "token_type_ids": 0},
                    0
                ),
                drop_remainder=True)
    return dataset, 79561622#count.numpy()
    

def _get_dataset_eval(dataset, batch_size, seq_length, num_skip=0):
    dataset = dataset.map( lambda record : _extract_fn_eval(record, seq_length)).prefetch(batch_size*1000)
    if num_skip > 0:
        dataset = dataset.skip(num_skip)
    count = dataset.reduce(0, lambda x, _: x + 1)
    dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=({
                    "q_id" : [],
                    "d_id" : [],
                    "input_ids": [seq_length],
                    "attention_mask": [seq_length],
                    "token_type_ids": [seq_length],
                    "len_gt_titles": []
                    }, []
                ),
                padding_values=({
                    "q_id": 0,
                    "d_id": 0,
                    "input_ids": 0,
                    "attention_mask": 0,
                    "token_type_ids": 0,
                    "len_gt_titles": 0
                    }, 0
                ),
                drop_remainder=False)
    return dataset, count.numpy()
    

def get_dataset(dataset_path, batch_size, seq_length, is_training_set=False, num_skip=0):

    dataset = tf.data.TFRecordDataset([dataset_path])
    if is_training_set:
        return _get_dataset_train(dataset, batch_size, seq_length, num_skip)
    else:
        return _get_dataset_eval(dataset, batch_size, seq_length, num_skip)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
