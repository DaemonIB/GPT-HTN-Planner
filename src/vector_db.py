import chromadb


class VectorDB:
    def __init__(self):
        # Create a chroma client
        self.client = chromadb.Client()

        # Create a collection
        self.collection = self.client.create_collection("task_nodes")

    def add_task_node(self, task_node):
        self.collection.upsert(documents=[task_node.task_name], ids=[task_node.node_name], metadatas=[task_node])

    def get_task_node(self, task_node):
        return self.collection.get(ids=[task_node.node_name])['metadatas'][0]

    def query_by_name(self, task_name):
        task_nodes = self.collection.query(query_texts=[task_name], n_results=1)
        return task_nodes['metadatas'][0]
