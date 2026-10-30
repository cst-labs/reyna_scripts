import truststore
truststore.inject_into_ssl()

from huggingface_hub import snapshot_download

path = snapshot_download("google/embeddinggemma-2")
print(path)
