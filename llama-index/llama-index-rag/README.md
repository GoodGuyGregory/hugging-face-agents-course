# LlamaIndex RAG

with RAG there are many steps during this tutorial we will focus on 5 main steps:

1. Loading: getting your data into your project. wether it's PDFs, CSV, or scraped HTML documentation with Beautiful Soup. LlamaHub has integrations if you're in a hurry.

2. Indexing: piping your data into something that makes searching easier, in most cases this is a Vector storage. Taking your daa and indexing and chunking them for that specific storage solution. 

3. Storing: After chunking and indexing your data with the correct variety of embeddings, or chunks it's time to store them into a solution that makes sense for LLM projects, ChromaDB, Pinecone etc are perfect locations to take your content for safe keeping.

4. Querying: This is the practical element of testing your storage solution with LlamaIndex provides structures we will cover to query your data 

5. Evaluation: Checking your work and determining it's recall/accuracy when prompting is the final portion of calling your RAG system complete.

## Loading Data:

LlamaIndex provides many ways to quickly plug into your data and begin working. Ideally having the data accessible can be challenging and LLamaIndex offers some solutions to this problem.

there are three main ways to load data into your RAG application code. 

`SimpleDirectoryReader`: this is a integration that allows for loading in files from a local directory
`LlamaParser`: This is a tool used to Parse PDFs
`LlamaHub`: when in doubt there are also likely options within the LlamaHub that can load any data you desire into your project. (if you can't find one make one )

### SimpleDirectoryReader

this tool is a quick way to access and create `Document` objects from any contents of a directory from your local project. 

**Setting up the SimpleDirectoryReader**

```python
from llama_index.core import SimpleDirectoryReader

# create an instance of the SimpleDirectoryReader
reader = SimpleDirectoryReader(input_dir="your/desired/path")

# collect your documents from the reader, by calling load_data()
documents = reader.load_data()
```

ideally with most RAG frameworks we chunk our documents and Embed them into a vector storage. The nomenclature, is a bit brand specific most of these companies are about renaming things and making them "their own". These small chunks are called **"Nodes"**

to create the Nodes, LlamaIndex leverages a proprietary Transformation tool. like any Framework, they are an `Embedding` and a `SentenceSplitter`. this is all included in an **IngestionPipeline** object.

**Create an IngestionPipeline**

```python
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline


# create the IngestionPipeline Object
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)


# collect the Nodes from our Document collection earlier
nodes = await pipeline.arun(documents=documents)

```

## Storing and Indexing Documents

if you're going to be storing a `Node` its best to store it inside of a `Vector Database` this is obvious with the modern architecture and strategy most Frameworks are built around due to its embedding success. However this will change our `IngestionPipeline` implementation and require us to add our chosen `vector_store`. Ideally Chroma is still the free tried and true option for this. importing the Vector store is pretty simple and adding it into our pipeline is easy.


**Add ChromaDB Dependency**

`uv add chromadb`

### Add ChromaDB into our Ingestion Pipeline

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# the PersistentClient is where the chroma database will store it's Nodes (Chunks/Embeddings of Document Objects)
db = chromadb.PersistentClient(path="./alfred_chroma_db")

# chroma collection represents the collection name
chroma_collection = db.get_or_create_collection("alfred")

# creates a vector_store object for our IngestionPipeline
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# now we add our additional vector storage solution we want to add.
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
    # add our new vector storage solution here: 
    vector_store=vector_store
)
```

Okay, now that the `vector_store` is added we can now embed our query with the `HuggingFaceEmbedding` to ensure all of hte vector Store entries feature the same embedding for similarity searching.

```python
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# chosen embedding model with the "BAAI/bge-small" model.
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# associate the embedding_model and the vector_store
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
```

our `index` is the vector store. The terminology can get fuzzy if you reference the documentation from Hugging Face's Agent's course. Ideally you would want to Query your index in a few different fashions there are a few listed on the documentation page.


* `as_retriever`: 
