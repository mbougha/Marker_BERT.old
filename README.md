# Marked_BERT

1. Upload the Marked_BERT.ipynb notebook to your Googel drive.
2. Run the cells and read the comments. 
3. The notebook is paramtered to do evaluation. 
4. The train dataset is too huge, I can't share it. 
5. The eval dataset is the test est from trec passage ranking task is under the data directory. You need to create a Google Cloud Storage (GCS) Bucket and put the .tf file in it so that the TPU can access it. 
6. I shared the checkpoint of the model on Google Drive. 
7. Don't forget to give the TPU the access permission to yout bucket, run it for the first time and you'll get an exception that contains the TPU name ==> go to your GCS page and add the TPU name in the permissions like explained [here](https://cloud.google.com/storage/docs/access-control/using-iam-permissions).
