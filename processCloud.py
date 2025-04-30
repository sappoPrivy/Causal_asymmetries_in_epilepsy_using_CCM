from google.cloud import storage

# EXAMPLE CODE

# Initialize the client
client = storage.Client()

# Reference your bucket
bucket_name = "chbmit-1.0.0.physionet.org"
bucket = client.get_bucket(bucket_name)

# List the files in the bucket
blobs = bucket.list_blobs()

# Print the names of the files
for blob in blobs:
    print(blob.name)
