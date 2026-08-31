import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
 
# ==========================================================
# CONFIGURATION
# ==========================================================
 
REGION = "ap-southeast-1"
HOST = "mc9xamsc1uiy9cxgnwt9.aoss.ap-southeast-1.on.aws"
INDEX_NAME = "kms-index-sop"
 
# ==========================================================
# AWS AUTHENTICATION
# ==========================================================
 
session = boto3.Session()
credentials = session.get_credentials()
 
auth = AWSV4SignerAuth(credentials, REGION, "aoss")
 
client = OpenSearch(
    hosts=[{"host": HOST, "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=120
)
 
 
# ==========================================================
# DELETE DOCUMENTS
# ==========================================================
 
def delete_documents(document_ids):
 
    if isinstance(document_ids, str):
        document_ids = [document_ids]
 
    for document_id in document_ids:
 
        print(f"\nSearching chunks for {document_id}...")
 
        response = client.search(
            index=INDEX_NAME,
            body={
                "size": 1000,
                "query": {
                    "term": {
                        "document_id": document_id
                    }
                }
            }
        )
 
        hits = response["hits"]["hits"]
 
        print(f"Found {len(hits)} chunks")
 
        for hit in hits:
 
            chunk_id = hit["_id"]
 
            client.delete(
                index=INDEX_NAME,
                id=chunk_id,
                
            )
 
            print(f"Deleted : {chunk_id}")
 
        print(f"{document_id} deleted successfully.")
 
 
# ==========================================================
# MAIN
# ==========================================================
 
if __name__ == "__main__":
 
    document_ids = [
        "SOP-124627",
        
        "CD-60393",
        "CD-60405",
        "CD-60406"
        
    ]
 
    delete_documents(document_ids)
 
    print("\nCompleted.")
 