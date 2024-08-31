# Import necessary libraries
import os
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chat_models import ChatOllama

def extract_and_store_graph(text):
    os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "neo4jpassword"
    
    # Initialize graph database
    graph = Neo4jGraph()
    
    # Initialize LLM model
    llm = ChatOllama(temperature=0, model="llama3")

    # Initialize transformer
    transformer = LLMGraphTransformer(llm=llm)

    # Extract graph data from text  
    documents = [Document(page_content=text)]
    
    # Debug print to check the documents
    for doc in documents:
        print(f"Document: {doc.page_content}")

    # Convert documents to graph documents
    print("Converting to graph doc...")
    try:
        graph_documents = transformer.convert_to_graph_documents(documents)
    except Exception as e:
        print(f"Error during conversion to graph documents: {e}")
        for doc in documents:
            print(f"Document causing error: {doc.page_content}")
        raise e
    
    # Debug print to check the graph_documents
    print("Graph Documents:")
    for graph_doc in graph_documents:
        print(graph_doc)

    # Store graph documents in Neo4j database
    graph.add_graph_documents(graph_documents)
    print("Graph data stored successfully!")

def main():
    # Example text
    text = """
    Marie Curie, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    She was, in 1906, the first woman to become a professor at the University of Paris.
    """
    
    # Extract and store graph data
    extract_and_store_graph(text)

if __name__ == "__main__":
    main()
