from pymilvus import MilvusClient, DataType

client = MilvusClient(
    uri="http://localhost:19530"
)

client.create_collection(
    collection_name="quick_setup",
    dimension=5
)

res = client.get_load_state(
    collection_name="quick_setup"
)

print(res)
