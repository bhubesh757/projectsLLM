#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install chromadb')
get_ipython().system('pip install langchain')
get_ipython().system('pip install numpy')
get_ipython().system('pip install -U langchain-openai')
get_ipython().system('pip install pydantic')
get_ipython().system('pip install shutil')
get_ipython().system('pip install openai==0.28')


# In[3]:


import chromadb


# In[15]:


get_ipython().system('pip install langchain==0.1.20')
get_ipython().system('pip install "langchain-openai==0.1.1"')
get_ipython().system('pip install "chromadb==0.4.22"')
get_ipython().system('pip install "pydantic<2"')
get_ipython().system('pip install tiktoken sentence-transformers')


# In[8]:


import pandas


# In[1]:


get_ipython().system('pip install chromadb')


# In[2]:


import chromadb


# In[92]:


import pandas


# In[10]:


pip install langchain langchain-core langchain-openai


# In[91]:


import os
import pandas as pd
import shutil
from dataclasses import dataclass

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from typing import List
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, NonNegativeInt
from langchain.prompts import PromptTemplate


# In[ ]:





# In[87]:


from langchain.vectorstores.chroma import Chroma 


# In[88]:


# Environment variables
OPENAI_API_KEY = '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'
os.environ['OPENAI_API_KEY']= OPENAI_API_KEY


# In[89]:


MODEL_NAME = 'openai/gpt-3.5-turbo'


# In[90]:


llm = OpenAI(model_name=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY , openai_api_base="https://openrouter.ai/api/v1")


# In[85]:


INSTRUCTION = "Generate a CSV file with at least 10 real estate listings."
SAMPLE_LISTING = """
Neighborhood: Green Oaks
Price: $800,000
Bedrooms: 3
Bathrooms: 2
House Size: 2,000 sqft

Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
"""


# In[9]:


class RealEstateListing(BaseModel):
    """
    A real estate listing.
    
    Attributes:
    - neighborhood: str
    - price: NonNegativeInt
    - bedrooms: NonNegativeInt
    - bathrooms: NonNegativeInt
    - house_size: NonNegativeInt
    - description: str
    - neighborhood_description: str
    """
    neighborhood: str = Field(description="The neighborhood where the property is located")
    price: NonNegativeInt = Field(description="The price of the property in USD")
    bedrooms: NonNegativeInt = Field(description="The number of bedrooms in the property")
    bathrooms: NonNegativeInt = Field(description="The number of bathrooms in the property")
    house_size: NonNegativeInt = Field(description="The size of the house in square feet")
    description: str = Field(description="A description of the property")
    neighborhood_description: str = Field(description="A description of the neighborhood.")  

class ListingCollection(BaseModel):
    """
    A collection of real estate listings.
    
    Attributes:
    - listings: List[RealEstateListing]
    """
    listings: List[RealEstateListing] = Field(description="A list of real estate listings")


# In[10]:


parser = PydanticOutputParser(pydantic_object=ListingCollection)
 


# In[11]:


# printing the prompt
prompt = PromptTemplate(
    template="{instruction}\n{sample}\n{format_instructions}\n",
    input_variables=["instruction", "sample"],
    partial_variables={"format_instructions": parser.get_format_instructions},
)

query = prompt.format(
    instruction=INSTRUCTION,
    sample=SAMPLE_LISTING,
)
print(query)


# In[12]:


llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0,
    api_key=OPENAI_API_KEY, # Use the API key
    openai_api_base="https://openrouter.ai/api/v1" # Specify the custom base URL
)


# In[13]:


response_msg = llm.invoke(query)
response_content = response_msg.content 


# In[15]:





# In[18]:


# create a dataframe from the response
result = parser.parse(response_content)
data_list = [item.dict() for item in result.listings]
df = pd.DataFrame(data_list)
df.head() 


# In[48]:


df


# In[20]:


df.to_csv('real_estate_listings.csv', index_label = 'id')
 


# In[32]:


import os
import shutil
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings # Corrected import for modern LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --- 1. DEFINE YOUR OPENROUTER SETTINGS ---
OPENROUTER_API_KEY = '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' # ⚠️ Make sure this is set!
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SITE_URL = "http://localhost:8080" # Required referrer for free models
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY

# --- THE FIX: Change CHROMA_PATH to use /tmp ---
CHROMA_PATH = "chroma3" # We are now saving in the system's temp folder
CSV_PATH = "real_estate_listings.csv"


# --- 2. CREATE THE CONFIGURED EMBEDDING FUNCTION ---
embedding_function = OpenAIEmbeddings(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    default_headers={
        "HTTP-Referer": DEFAULT_SITE_URL
    }
)



# --- 3. LOAD, SPLIT, AND SAVE (No changes here) ---
df = pd.read_csv(CSV_PATH)
documents = []

for index, row in df.iterrows():
    documents.append(Document(page_content=row['description'], metadata={'id': str(index)}))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(chunks)} chunks.")


if chunks:
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

# --- 4. SAVE TO THE NEW /tmp PATH ---
# We still clear it first, just to be safe

if os.path.exists(CHROMA_PATH):
    print(f"Removing old database from {CHROMA_PATH}...")
    shutil.rmtree(CHROMA_PATH)

# We don't even need chmod. /tmp is designed to be writable.
os.makedirs(CHROMA_PATH, exist_ok=True)
print(f"Created new directory at {CHROMA_PATH}.")

print("Attempting to save to Chroma...")
db = Chroma.from_documents(
    chunks,
    embedding_function, # <-- Use the pre-configured instance
    persist_directory=CHROMA_PATH
)
db.persist()
print(f"Successfully saved {len(chunks)} chunks to {CHROMA_PATH}.")


# In[38]:


import os
import shutil
import tempfile
import pandas as pd
from dataclasses import dataclass

# LangChain (modern imports)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ------------------------------------------
# 1. OPENROUTER / OPENAI CONFIG
# ------------------------------------------

# ❗ Load API Key from environment variables
OPENROUTER_API_KEY = "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"

# if OPENROUTER_API_KEY is None:
#     raise ValueError("ERROR: Set environment variable OPENROUTER_API_KEY before running.")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SITE_URL = "http://localhost:8080"

# LangChain uses the OpenAI key, so map it here:
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY

# ------------------------------------------
# 2. CHROMA PERSIST DIRECTORY (REAL TEMP FOLDER)
# ------------------------------------------

CHROMA_PATH = os.path.join("chroma1")

print("Chroma DB Path:", CHROMA_PATH)

# Completely remove old DB
if os.path.exists(CHROMA_PATH):
    print("Removing old DB...")
    shutil.rmtree(CHROMA_PATH)

# Recreate with full permissions
os.makedirs(CHROMA_PATH, exist_ok=True)
os.chmod(CHROMA_PATH, 0o777)

print("Created writable directory:", CHROMA_PATH)

# ------------------------------------------
# 3. LOAD CSV DOCUMENTS
# ------------------------------------------

CSV_PATH = "real_estate_listings.csv"
df = pd.read_csv(CSV_PATH)

documents = []
for index, row in df.iterrows():
    documents.append(
        Document(
            page_content=row["description"],
            metadata={"id": str(index)}
        )
    )

print(f"Loaded {len(documents)} documents.")

# ------------------------------------------
# 4. SPLIT DOCUMENTS INTO CHUNKS
# ------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(chunks)} chunks.\n")

# Show one sample chunk
if chunks:
    sample = chunks[10]
    print("Sample chunk text:")
    print(sample.page_content)
    print("Metadata:", sample.metadata)

# ------------------------------------------
# 5. EMBEDDINGS SETUP (OpenRouter / OpenAI)
# ------------------------------------------

embedding_function = OpenAIEmbeddings(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    default_headers={"HTTP-Referer": DEFAULT_SITE_URL},
)

# ------------------------------------------
# 6. SAVE THE CHUNKS INTO CHROMA DB
# ------------------------------------------

print("\nSaving chunks into ChromaDB...")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory=CHROMA_PATH,
)

db.persist()
print(f"SUCCESS: Saved {len(chunks)} chunks to {CHROMA_PATH}")


# In[40]:


import chromadb


# In[78]:


import os
# from langchain.prompts import ChatPromptTemplate
# New, correct import
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma


def predict_response(query_text, PROMPT_TEMPLATE):
    # Re-initialize embedding_function with OpenRouter settings
    embedding_function = OpenAIEmbeddings(
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": DEFAULT_SITE_URL
        }
    )
    # CHROMA_PATH is defined globally earlier in the notebook
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print(f"Generated Prompt:\n{prompt}")

        # Re-initialize ChatOpenAI with OpenRouter settings
        model = ChatOpenAI(
            temperature=0.7,
            api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL
        )
        response_text = model.invoke(prompt).content
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text} \nSources: {sources}"
#         print(formatted_response)
        print("\n=============================")
        print("🔹 RESPONSE")
        print("=============================")
        print(formatted_response)

        print("\n=============================")
        print("🔹 SOURCES (FULL DOCUMENTS)")
        print("=============================")

    # Print full source details
        for doc, score in results:
            print("\n--- Source Document ---")
            print(f"ID: {doc.metadata.get('id')}")
            print(f"Relevance Score: {score}")
            print("Full Metadata:", doc.metadata)
            print("Content:")
            print(doc.page_content)
            print("------------------------")
        


# In[79]:


query_text = "Find luxury homes priced above $2 million that feature outdoor kitchens, swimming pools, and large entertainment-ready backyards."


# In[80]:


BASIC_PROMPT_TEMPLATE ="""
Based on the following context:

{context}

Answer the question : {question}
"""


# In[81]:


predict_response(query_text, BASIC_PROMPT_TEMPLATE)


# In[76]:


AUGMENT_PROMPT_TEMPLATE ="""
Based on the following context:

{context}

---

craft a response that not only answers the question {question}, but also ensures that your explanation is distinct, captivating, and customized to align with the specified preferences. This involves subtly emphasizing aspects of the property that align with what the buyer is looking for.
"""


# In[77]:


predict_response(query_text, AUGMENT_PROMPT_TEMPLATE)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




