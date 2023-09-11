import json
from video_indexer import VideoIndexer

CONFIG = {
    'SUBSCRIPTION_KEY': '<your key>',
    'LOCATION': 'trial',
    'ACCOUNT_ID': '<your ID>'
}

vi = VideoIndexer(
    vi_subscription_key=CONFIG['SUBSCRIPTION_KEY'],
    vi_location=CONFIG['LOCATION'],
    vi_account_id=CONFIG['ACCOUNT_ID']
)


# # Creating Indexes for Videos 


import os
import time
import datetime
_path='<Path>'




video_list=[]
info_list=[]
for file in os.listdir(_path):
    video_id = vi.upload_to_video_indexer(input_filename=_path+file,video_name=file+'treated',video_language='English')
    video_list.append(video_id)
    time.sleep(120)



info_list=[]
for i in video_list:
    info = vi.get_video_info(i,video_language='English')
    info_list.append(info)





import pandas as pd
df_transcript_final=pd.DataFrame()

for i in info_list:
    r = json.dumps(i) 
    loaded_r = json.loads(r)
    r_refined=loaded_r['videos'][0]['insights']['transcript']
    df_cleaned=pd.DataFrame.from_dict(r_refined)
    df_cleaned_final= pd.concat([df_cleaned.drop(['instances'], axis=1), df_cleaned['instances'].apply(pd.Series)], axis=1)
    df_cleaned_final.columns=['id','text','confidence','speakerId','language','json_extract']
    df_cleaned_final= pd.concat([df_cleaned_final.drop(['json_extract'], axis=1), df_cleaned_final['json_extract'].apply(pd.Series)], axis=1)
    df_cleaned_final['file_name']=loaded_r['name']
    df_transcript_final=df_transcript_final.append(df_cleaned_final)
    del df_cleaned, df_cleaned_final
    



import os
import openai
import chromadb
from langchain.llms import AzureOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import json
import io
import pandas as pd
import os
import openai
from urllib.parse import quote
import sys
from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI, AzureOpenAI
from langchain.chains import (
    ConversationalRetrievalChain,
    RetrievalQAWithSourcesChain,
    LLMChain,
)
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

from langchain.agents import create_sql_agent, create_csv_agent
from langchain import SQLDatabaseChain
from langchain.llms import AzureOpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
#from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool, load_tools
from langchain.utilities import PythonREPL

##Azure Open AI
os.environ["AZURE_OPEN_AI_API_TYPE"] = "azure"
os.environ["AZURE_OPEN_AI_ENDPOINT"] = "<>"
os.environ["AZURE_OPEN_AI_API_VERSION"] = "<>"
os.environ["AZURE_OPEN_AI_API_KEY"] = "<>"
os.environ["AZURE_OPEN_AI_DEPLOYMENT_MODEL"] = "<>"
os.environ["AZURE_OPEN_AI_PROMPT_TEMPERATURE"] = "0"

openai.api_type = os.environ["AZURE_OPEN_AI_API_TYPE"]
openai.api_base = os.environ["AZURE_OPEN_AI_ENDPOINT"]
openai.api_version = os.environ["AZURE_OPEN_AI_API_VERSION"]
openai.api_key = os.environ["AZURE_OPEN_AI_API_KEY"]

os.environ["OPENAI_API_TYPE"] = os.environ["AZURE_OPEN_AI_API_TYPE"]
os.environ["OPENAI_DEPLOYMENT_MODEL"] = os.environ["AZURE_OPEN_AI_DEPLOYMENT_MODEL"]
os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPEN_AI_API_VERSION"]
os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPEN_AI_ENDPOINT"]
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPEN_AI_API_KEY"]


# In[7]:


from langchain.document_loaders import CSVLoader

df_transcript_final.to_csv('final_text.csv')

# load the document as before
loader = CSVLoader('./final_text.csv')
documents = loader.load()


# In[8]:


import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


# In[9]:


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)


# In[10]:


from langchain.vectorstores import FAISS
vectordb = FAISS.from_documents(
      documents,
      embedding=HuggingFaceEmbeddings()
    )


# In[13]:


from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=AzureOpenAI(deployment_name = "<name>", 
                          model_name = "<name>",
                          temperature = 0, 
                          verbose=True),
    retriever=vectordb.as_retriever(search_kwargs={'k': 7}),
    return_source_documents=True
)


# In[16]:


# we can now execute queries against our Q&A chain
prompt='Revert with an elaborate answer along with the name of the file in question and the timestamp'
result = qa_chain({'query': 'What are the main messages from the Drinking and driving video and the warehouse video ?'+prompt})
print(result['result'])


