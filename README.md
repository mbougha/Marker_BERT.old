# Marked_BERT

1. Upload the Marked_BERT.ipynb notebook to your Googel drive.
2. Run the cells and read the comments. 
3. The notebook is paramtered to do evaluation. 
4. The train dataset is too huge, I can't share it. 
5. The eval dataset is the test est from trec passage ranking task is under the data directory. You need to create a Google Cloud Storage (GCS) Bucket and put the .tf file in it so that the TPU can access it. 
6. I shared the checkpoint of the model on Google Drive. 
7. Don't forget to give the TPU the access permission to yout bucket, run it for the first time and you'll get an exception that contains the TPU name ==> go to your GCS page and add the TPU name in the permissions like explained [here](https://cloud.google.com/storage/docs/access-control/using-iam-permissions).

# Collection Cord-19 example
1. Use the COVIDEX_1.ipynb to extract data from the anserini Index, execute up to the section `` Create Run document level from passage level``  ==> get some files 
    a) topics_rnd1.tsv: containing the queries
    b) COVID_run.tsv : the run
    c) CORD19_run_1_docs.tsv: the documents text file
2. Use the script convert_cord19_1.py to create a run file ready for the markers. U can execute it on osirim here's a run.sh file:

```
#!/bin/sh
#SBATCH --job-name=prepare_inference_data
#SBATCH --cpus-per-task=5
#SBATCH --partition=24CPUNodes
#SBATCH --output=./outputs/cord19_run.out    # Standard output and error log
 

OUT_DIR=/path/to/output/dir  # don't add / at the end

srun singularity exec /logiciels/containerCollections/CUDA10/tf2-NGC-19-11-py3.sif  $HOME/tf2Env/bin/python3.6 "$HOME/workspace/MarkedBERT/convert_cord19_1.py" \ # path to the script
	--output_folder $OUT_DIR \
	--queries_path $OUT_DIR/topics_rnd1.tsv \ # queries
	--run_path $OUT_DIR/CORD_run_1.tsv \  #run
	--collection_path $OUT_DIR/CORD19_run_1_docs.tsv \ #docs
	--num_eval_docs 1000 \ # top 1000
  --set_name cord19  # set name just to get an output file with this name 

```

3. Now you can mark, the script ``` marker_cord19_1.py``` does all the work, the data dir is the output dir of the step 2 :
```
#!/bin/sh
#SBATCH --job-name=mark_inference_data
#SBATCH --cpus-per-task=10
#SBATCH --partition=24CPUNodes
#SBATCH --output=./outputs/cord19_mark_base.out    # Standard output and error log

DATA_DIR=/projets/iris/PROJETS/lboualil/workdata/msmarco-passage/MarkedBERT_large/Cord19/round1
OUT_DIR=/projets/iris/PROJETS/lboualil/workdata/msmarco-passage/MarkedBERT_large/My_MsMarco_TFRecord/trec/cord19/round1/base


srun singularity exec /logiciels/containerCollections/CUDA10/tf2-NGC-19-11-py3.sif  $HOME/tf2Env/bin/python3.6 "$HOME/workspace/MarkedBERT/marker_cord19_1.py" \
	--output_dir $OUT_DIR \ # make sure to use different output dirs for different strategies because the output file will have the same name, you can change the set_name to make a difference between them (eg: set_name cord9_base, set_name cord19_un_mark_pair ... )
	--data_path $DATA_DIR/run_cord19_doc.tsv \ # the file result of step 2
	--handle split \ # use overlapping passages
	--strategy base \ # change here strategy of marking : base, un_mark_pass, un_mark_pair, mu_mark_pass, mu_mark_pair
	--set_name cord19

```

4. Now u have the dataset_*.tf file that u can use in a colab with bert. Use the updated colab file ```Marked_BERT.ipynb```. I don't know if you changed something in your version, if so repport them in this version too: I added a new paramter strategy, I changed the imports , And in the main I left a comment # modified section .


# If you use another corpus, you need to see the data processors understand how it works, I gave you all the converters and markers for all datasets I used : for robust04 it is similar to cord19 but for trec the files needed for convert_*.py are different read the parameters. I can't explain all of them but if you read the convert dataset methods in each processor u can understand the format of each file.  
