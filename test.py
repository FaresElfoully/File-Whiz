import nest_asyncio
nest_asyncio.apply()
from llama_parse import LlamaParse
api_key = "llx-ymOYI3cuZOpM4K3OfKbxZEu6PRZqbnTo8A0r1ja1zGqI5ScM"
parser = LlamaParse(api_key=api_key,result_type="markdown")
document = parser.load_data("employee_data.xlsx")
import os
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set up Groq API Key
os.environ['GROQ_API_KEY'] = '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'

# Set up a local embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize the Groq LLM with a Llama model
try:
    llm = Groq(
        model="llama-3.2-90b-vision-preview", 
        api_key=os.environ.get('GROQ_API_KEY')
    )

    # Configure Settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Create the index and query engine
    index = VectorStoreIndex.from_documents(
        document, 
        embed_model=embed_model
    )
    query_engine = index.as_query_engine()

    # Perform query
    response = query_engine.query("How many vacation days do I have?")
    print(str(response))

except Exception as e:
    print(f"An error occurred: {e}")
