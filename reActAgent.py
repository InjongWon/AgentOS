# Learn how to create tools in langraph
# work with different types of messages, toolMessages, baseMessage
# create a robust ReAct Agent

from typing import TypedDict, List, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

# Annotated - pupose is to preserve the rule whether its type or the messages like using add_messages.
# Sequence - just iterative data structure to use
# SystemMessage - message to the LLM
# BaseMessage - Foundational Messages is all AIMessage,ToolMessage, HumanMessage, SystemMessage are part of BaseMessage.
load_dotenv()

# add_message: merge new data into existing states without overwriting.

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def add(a:int, b:int)->int:
    "this adds two int together"
    return a+b

tools = [add]
model = ChatOpenAI(model = "gpt-4o", temperature = 0.5).bind_tools(tools)
# now LLM has access to all tools

def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content='you are my AI assistant, please answe my question best in depth')
    response = model.invoke([system_prompt]+ list(state['messages']))
    
    return {'messages':[response]}


def should_continue(state:AgentState):
    messages= state['messages']
    
    last_message=messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node('our_agent', model_call)

tool_node = ToolNode(tools)
graph.add_node('tools', tool_node)

graph.set_entry_point('our_agent')
graph.add_conditional_edges(
    'our_agent',
    should_continue,
    {
        'continue':'tools',
        'end':END
    },
    
)
graph.add_edge('tools', 'our_agent')
agent = graph.compile()

inputs = {'messages':[('user', 'ADD 3+ 4')]}
def print_response(stream):
    for s in stream:
        msg = s['messages'][-1] #last one
        if isinstance(msg, tuple):
            print(msg)
        else:
            msg.pretty_print()

stream_output = agent.stream(inputs, stream_mode = 'values')
print_response(stream_output) 