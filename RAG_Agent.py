import doctest
from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages.tool import tool_call
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import  add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

llm = ChatOpenAI(
    model = "gpt-4o", temperature =0.0
)

embedding = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)

pdf_path = " "

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at: {pdf_path}")


pdf_loader = PyPDFLoader(pdf_path)
docs = pdf_loader.load()

try:
    pages = pdf_loader.load()
    print(f"Loaded {len(pages)} pages from {pdf_path}")
except Exception as e:
    raise ValueError(f"Error loading PDF: {e}")

#chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = " "
collection_name = "pdf_collection"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        document = pages_split,
        embedding = embedding,
        persis_directory = persist_directory,
        collection_name = collection_name
    )
    print(f"Vectorstore created and documents indexed")
except Exception as e:
    raise ValueError(f"Error creating vectorstore: {e}")

try:
    vectorstore = Chroma(
        persist_directory = persist_directory,
        embedding_function = embedding,
        collection_name = collection_name
    )
    print(f"created ChromDB for vector store")

except Exception as e:
    print(f"Error cretaing ChromDB:{str(e)}")
    raise

    
# retriever 
retriever = vectorstore.as_retreiver(
    search_type = "similarity",
    search_kwargs = {"k":5} # top 5 most similar chunks
    
)

@tool 
def retrive_tool(query:str)-> str:
    """
    Tool searches return the information from the stock market financial documents
    """
    docs = retriever.invoke(query)
    if not docs:
        raise ValueError(f"No documents found for query: {query}")

    results = []
    for i, docs in enumerate(docs, start=1):
        results.append(f"Document {i}:\n{docs.page_content}")
        
    return "\n\n".joing(results)

tools = [retrive_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage], add_messages]
    
def should_continue(state:AgentState):
    """ check if the last message contains a tool call"""
    last_message = state['messages'][-1]
    return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0

system_prompt = f"""
You are a helpful assistant that can answer questions about the stock market financial documents.
You have access to the following tools:
{tools}

You will be given a question and you need to answer it based on the information from the stock market financial documents.
If you don't know the answer, you can use the tool to search the information from the stock market financial documents.
If you have the answer, you can return the answer directly.
If you don't have the answer, you can use the tool to search the information from the stock market financial documents.
If you have the answer, you can return the answer directly.
Please always cite the specific parts of the documents

"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

# LLM Agent
def call_llm(state:AgentState)->AgentState:
    messages= list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    
    return{'messages':[message]}


# retriver agent

def take_action(state:AgentState):
    "Execute tool calls from the last message"
    tool_calls = state['messages'][-1].tool_calls
    
    result = []
    for tool in tool_calls:
        print(f"Calling Tool:{tool['name']} with query: {tool['args'].get('query', 'No Query provided')}")
        
        if not tool['name'] in tools_dict:
            print(f"\nTool:{t['name']} does not exist")
            result = " Incorrect toole name, please retry again and select the correct tool "
        
        else:
            result = tools_dict[tool['name']].invoke(tool['args'].get('query', ' '))
            print(f"Tool Result:{result}")

        result.append(ToolMessage(tool_call_id = tool['id'], name = tool['name'], content = str(result)))
    
    return {'messages':result}

graph = StateGraph(AgentState)
graph.add_node('call_llm', call_llm)
graph.add_node('retriever', take_action)

graph.add_edge(START, 'call_llm')
graph.add_edge('call_llm', 'take_action')

graph.add_conditional_edges(
    'llm',
    should_continue,
    {True:'retriver', False:END}
    
)


graph.set_entry_point('call_llm')
graph.set_finish_point('take_action')

app = graph.compile()
