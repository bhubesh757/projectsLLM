#!/usr/bin/env python
# coding: utf-8

# In[29]:


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
from langchain.chains import RetrievalQA

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


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


# In[30]:


import openai 


# In[31]:


from langchain import LLMChain
from langchain.chains.question_answering import load_qa_chain 


# In[32]:


import os

OPENAI_API_KEY = "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"
OPENAI_BASE = "https://openrouter.ai/api/v1"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_BASE


# In[33]:


MODEL_NAME = 'openai/gpt-3.5-turbo'


# In[34]:


llm = OpenAI(model_name=MODEL_NAME, 
             temperature=0, 
             api_key=OPENAI_API_KEY , 
             openai_api_base="https://openrouter.ai/api/v1")


# In[35]:


INSTRUCTION = "Generate a CSV file with at least 20 real estate listings."
SAMPLE_LISTING = """
Neighborhood: Green Oaks
Price: $800,000
Bedrooms: 3
Bathrooms: 2
House Size: 2,000 sqft

Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
""" 


# In[36]:


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


# In[37]:


parser = PydanticOutputParser(pydantic_object=ListingCollection)


# In[38]:


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


# In[39]:


llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0,
    api_key=OPENAI_API_KEY, # Use the API key
    openai_api_base="https://openrouter.ai/api/v1" # Specify the custom base URL
)


# In[40]:


response_msg = llm.invoke(query)
response_content = response_msg.content 


# In[41]:


# create a dataframe from the response
result = parser.parse(response_content)
data_list = [item.dict() for item in result.listings]
df = pd.DataFrame(data_list)
df.head() 


# In[42]:


df.to_csv('real_estate_listings.csv', index_label = 'id')
 


# In[43]:


df


# In[ ]:





# ## Vector Database . 

# In[44]:


# --- THE FIX: Change CHROMA_PATH to use /tmp ---
CHROMA_PATH = "chroma2" # We are now saving in the system's temp folder
CSV_PATH = "real_estate_listings.csv"


# --- 2. CREATE THE CONFIGURED EMBEDDING FUNCTION ---
embedding_function = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE,
)


# --- 3. LOAD, SPLIT, AND SAVE (No changes here) ---
loader = CSVLoader(file_path="real_estate_listings.csv")
docs = loader.load()


text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)

chunks = text_splitter.split_documents(docs)
print(f"Split {len(docs)} documents into {len(chunks)} chunks.")


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


# In[ ]:





# In[45]:


# # SEMANTIC SEARCH QUERY
# user_query = "Recommend the house with the beach "

# results = db.similarity_search(user_query, k=5)

# for i, r in enumerate(results, 1):
#     print(f"\n--- Result {i} ---")

#     # Print metadata (raw title-level info)
#     for key, value in r.metadata.items():
#         print(f"{key}: {value}")


# In[ ]:





# In[ ]:





# ## Step 2 
# 

# In[46]:


from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load CSV
loader = CSVLoader(file_path="real_estate_listings.csv")
docs = loader.load()

# Split long descriptions if needed
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = splitter.split_documents(docs)

# Embeddings + ChromaDB
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(split_docs, embeddings) 


# In[ ]:





# In[48]:


from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# SEMANTIC SEARCH QUERY
user_query = "Recommend the house with ocean with coastal  "

# Retrieve top 5 similar documents
results = db.similarity_search(user_query, k=5)

for i, r in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(r.page_content)


# ##  Semantic Search Using RAG 

# In[ ]:





# In[52]:


from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# LLM
llm = OpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=500
)

# Build Retriever
retriever = db.as_retriever(search_kwargs={"k": 5})

# RAG Chain
rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

query = "Recommend the house with ocean view  "

use_chain_helper = False
if use_chain_helper:
    rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    print(rag.run(query))
else:
    similar_docs = db.similarity_search(query, k=5)
    prompt = PromptTemplate(
        template="{query}\nContext: {context}",
        input_variables=["query", "context"],
    )
    chain = load_qa_chain(llm, prompt = prompt, chain_type="stuff")
    print(chain.run(input_documents=similar_docs, query = query)) 


# In[ ]:





# In[ ]:





# ## Personalized Recommendation 

# In[26]:


from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI


# In[ ]:





# In[63]:



# ----------------------------------------------------
# 1. PERSONAL QUESTIONS
# ----------------------------------------------------
personal_questions = [
    "What is your ideal number of bedrooms?",
    "What kind of neighborhood do you prefer?",
    "What is your maximum budget?",
    "Which amenities matter most to you?",
    "Do you prefer urban, suburban, or semi-urban areas?"
]

# ----------------------------------------------------
# 2. HARD-CODED USER ANSWERS (YOU CAN CHANGE THESE)
# ----------------------------------------------------
answers = [
    "I’m looking for a spacious 3-bedroom home.",
    "I prefer like a art based and with island with city view",
    "My budget is around 2800000 ",
    " nearby parks, boutique shops",
    "urban "
]


# ----------------------------------------------------
# 3. BUILD CHAT HISTORY (SIMULATE A FULL CONVERSATION)
# ----------------------------------------------------
history = ChatMessageHistory()

history.add_user_message(
    f"You are a real estate AI assistant. Ask the user {len(personal_questions)} personalization questions."
)

for q, a in zip(personal_questions, answers):
    history.add_ai_message(q)       # AI question
    history.add_user_message(a)     # User answer


# ----------------------------------------------------
# 4. CREATE A WORKING SUMMARIZATION MEMORY (THE FIX)
# ----------------------------------------------------
llm_summary = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=200
)

memory = ConversationSummaryMemory(
    llm=llm_summary,
    chat_memory=history,
    memory_key="summary",
    input_key="question",
    return_messages=True
)

# IMPORTANT: Trigger memory to summarize your Q/A
memory.save_context(
    {"question": "Summarize the user's home preferences."},
    {"answer": "Sure, here is the summary."}
)

# Load final summary
summary = memory.load_memory_variables({"question": "Summarize preferences"})

summary_text = memory.predict_new_summary(
    existing_summary="", 
    messages=history.messages
)


# summary_text = summary["summary"]

print("\n\n=== USER PREFERENCE SUMMARY ===")
print(summary_text)


# ----------------------------------------------------
# 5. PERSONALIZED RAG PROMPT
# ----------------------------------------------------
prompt = PromptTemplate(
    template="""
You are a smart real estate advisor.

USER PREFERENCES SUMMARY:
{summary}

RETRIEVED PROPERTY INFORMATION:
{context}

USER QUESTION:
{question}

Answer in a friendly, helpful tone (max 5 sentences).
""",
    input_variables=["summary", "context", "question"]
)

chain_type_kwargs = {"prompt": prompt}


# ----------------------------------------------------
# 6. BUILD FINAL PERSONLIZED RAG SYSTEM
# ----------------------------------------------------
personalized_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=400),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    combine_docs_chain_kwargs=chain_type_kwargs,
    memory=memory
)


# ----------------------------------------------------
# 7. FINAL QUERY (THIS IS WHERE RETRIEVAL HAPPENS)
# ----------------------------------------------------
final_query =  f"Recommend the best property that matches these preferences: {summary_text}"


# ----------------------------------------------------
# 8. RUN AND PRINT FINAL RECOMMENDATION
# ----------------------------------------------------
response = personalized_chain({
    "question": final_query,
    "chat_history": []
})

print("\n\n=== FINAL PERSONALIZED RECOMMENDATION ===")
print(response["answer"])




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




