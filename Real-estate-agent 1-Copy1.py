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


# In[ ]:





# In[15]:


get_ipython().system('pip install langchain==0.1.20')
get_ipython().system('pip install "langchain-openai==0.1.1"')
get_ipython().system('pip install "chromadb==0.4.22"')
get_ipython().system('pip install "pydantic<2"')
get_ipython().system('pip install tiktoken sentence-transformers')


# In[8]:


import pandas


# In[ ]:





# In[ ]:





# In[1]:


import pandas


# In[10]:


pip install langchain langchain-core langchain-openai


# In[3]:


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


import os
import openai

os.environ["OPENAI_API_KEY"] = "YOUR API KEY"
os.environ["OPENAI_BASE_URL"] = "https://openai.vocareum.com/v1"

openai.api_key = "YOUR API KEY"
openai.api_base = "https://openai.vocareum.com/v1"


prompt = """
Generate 10 synthetic real estate listings.
Return ONLY valid CSV format (no backticks, no commentary).
Columns:
Neighborhood,Price,Bedrooms,Bathrooms,House Size,Description

Rules:
- Bedrooms must be an integer between 1 and 5.
- Bathrooms must be an integer between 1 and 4.
- Price must be a realistic positive number.
- House Size must be a positive number in square feet.
- Add a proper in detail description for each estate.
- There should not be any None/Null/Void Cell in the CSV
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.5
)

csv_text = response.choices[0].message.content.strip()

# Save directly as CSV file
with open("listings.csv", "w", encoding="utf-8") as f:
    f.write(csv_text)

print("CSV generated successfully: listings.csv")


# In[ ]:





# In[4]:


from langchain.vectorstores.chroma import Chroma 


# In[ ]:





# In[5]:


# Environment variables
OPENAI_API_KEY = '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'
os.environ['OPENAI_API_KEY']= OPENAI_API_KEY


# In[6]:


MODEL_NAME = 'openai/gpt-3.5-turbo'


# In[7]:


llm = OpenAI(model_name=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY , openai_api_base="https://openrouter.ai/api/v1")


# In[8]:


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


# In[ ]:





# In[14]:


# create a dataframe from the response
result = parser.parse(response_content)
data_list = [item.dict() for item in result.listings]
df = pd.DataFrame(data_list)
df.head() 


# In[15]:


df


# In[16]:


df.to_csv('real_estate_listings.csv', index_label = 'id')
 


# In[17]:


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

# Working


# In[38]:





# In[23]:


def predict_response(query_text, PROMPT_TEMPLATE):
    embedding_function = OpenAIEmbeddings(
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": DEFAULT_SITE_URL},
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.7:
        print("No matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(
        temperature=0.7,
        api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
    )

    response = model.invoke(prompt).content

    print("\n=============================")
    print("🔹 RESPONSE")
    print("=============================")
    print(response)

    print("\n=============================")
    print("🔹 SOURCES")
    print("=============================")

    for doc, score in results:
        print(f"ID: {doc.metadata.get('id')} | Score: {score}")
        print(doc.page_content)
        print("------------------------")


# In[ ]:





# In[25]:


###############################################################
# CLEAN STEP 1 — SEMANTIC SEARCH OF LISTINGS
###############################################################

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

# Build retriever
retriever = db.as_retriever(search_kwargs={"k": 5})

query = "Find a safe neighborhood with 3 bedrooms and a good backyard."

results = retriever.get_relevant_documents(query)

print("=============================================")
print("🔍 STEP 1 — SEMANTIC SEARCH RESULTS")
print("=============================================")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Listing ID: {doc.metadata.get('id')}")
    print(f"Description: {doc.page_content}")
    print("---------------------------------------------")


###############################################################
# CLEAN STEP 2 — PERSONALIZATION (Conversation Summary Memory)
###############################################################

from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

personal_questions = [
    "What is your ideal number of bedrooms?",
    "What kind of neighborhood do you prefer?",
    "What is your maximum budget?",
    "Which amenities matter most to you?",
    "Do you prefer urban, suburban, or semi-urban areas?"
]

answers = [
    "I want 3 bedrooms.",
    "A quiet and family-friendly neighborhood.",
    "My maximum budget is 90 lakhs.",
    "Nearby schools, park, and parking.",
    "I prefer suburban areas."
]

history = ChatMessageHistory()

# Simulated conversation
history.add_user_message("Start preference collection.")

for q, a in zip(personal_questions, answers):
    history.add_ai_message(q)
    history.add_user_message(a)

# Summarize preferences
summary_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
)

memory = ConversationSummaryMemory(
    llm=summary_llm,
    chat_memory=history,
    memory_key="summary",
    input_key="question",
    buffer="Summarize clearly:"
)

memory.load_memory_variables({"question": "Summarize my home preferences."})
summary_text = memory.buffer

print("\n=============================================")
print("📝 STEP 2 — USER PREFERENCE SUMMARY")
print("=============================================")
print(summary_text)


###############################################################
# CLEAN STEP 3 — PERSONALIZED RAG RECOMMENDATION
###############################################################

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

personalized_prompt = PromptTemplate(
    template="""
You are a smart real estate advisor.

User Preferences Summary:
{summary}

Retrieved Property Information:
{context}

User Question:
{question}

Provide a friendly, helpful recommendation (max 5 sentences).
Explain clearly how the listing matches the user's preferences.
""",
    input_variables=["summary", "context", "question"]
)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key=OPENAI_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    ),
    retriever=retriever,
    chain_type="stuff",
    combine_docs_chain_kwargs={"prompt": personalized_prompt},
    memory=memory
)

final_query = "Recommend the best house for me."

response = personalized_chain({"question": final_query,
                             "chat_history":[]})

print("\n=============================================")
print("🏡 STEP 3 — PERSONALIZED RAG RECOMMENDATION")
print("=============================================")
print(response["answer"])


###############################################################
# CLEAN STEP 3B — Show Sources Used
###############################################################

print("\n=============================================")
print("📚 SOURCES (IDs & Text Used for Recommendation)")
print("=============================================")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Source {i} ---")
    print(f"Listing ID: {doc.metadata.get('id')}")
    print(doc.page_content)
    print("---------------------------------------------")


# In[ ]:





# In[20]:





# In[ ]:





# In[ ]:





# In[78]:





# In[79]:





# In[80]:





# In[ ]:





# In[76]:





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





# In[ ]:




