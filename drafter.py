# company need to create AI agnetic system that can speed up drafting documents, emails. 
# Agnetic AI systems should have human AI collaboration. (human should be able to provide continuous feed back and the AI agent should stop when the human is satisfied with the draft)
from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages.tool import tool_call
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import  add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    


@tool
def update(content:str)->str:
    """update the document content
    
    
    """
    global document_content
    
    document_content = content  # updating global variable and update with the passed in new document that we will modify
     
    return f"Document updated well: the current document is: \n {document_content}"

@tool
def save(filename:str)->str:
    """_summary_

    Args:
        filename: will be a text file

    Returns:
        str: _description_
    """
    global document_content
    if not filename.endswith('.txt'):
        filename= f"{filename}.txt"
    
    try:
        with open(filename, "w") as file:
            file.write(document_content)
        
        print(f"Documnet has been saved to: {filename}")
        return f"document saved {filename}"
    except Exception as e:
        return f"error saving this document name:{str(e)}"
    
tools = [save, update]
model = ChatOpenAI(model = "gpt-4o").bind_tools(tools) # basically calling llm to use created defined tools
# now the agent is a node in the graph

def our_agent(state:AgentState)-> AgentState:
    system_prompt=f"""
    You ar drafter, helpful writing assistant you are going to help the user update the draft document
    
    1. if the user wants to update or modify the content use tool 'update' 
    2. if the user want to save use tool 'save'
    then return the final document content:
    {document_content}
    
    """
    
    if not state['messages']:
        user_input = "I'm readyto help you update the draft"
        user_msg = HumanMessage(content=user_input)
    else:
        user_input = input("what do you want to do with the documnet")
        print(f"user input: {user_input}")
        user_msg= HumanMessage(content=user_input)
        
    all_messages = [system_prompt] + list(state['messages']) + [user_msg]
    
    response = model.invoke(all_messages)
    print(f"AI: {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f" Using Tools: {[tc['name'] for tc in response.tool_calls]}")
    
    return {'messages': list(state['messages']) + [user_msg, response]} 

def decide_next_step(state:AgentState)->str:
    """ Determine continue to new draft or end"""
    messages = state['messages']
    if not messages:
        return "continue"
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and 'saved' in message.content.lower() and 'document' in message.content.lower():
            return 'end' 
    return "continue"
        
graph = StateGraph(AgentState)
graph.add_node('our_agent',our_agent)
graph.add_node('tools', ToolNode(tools))


graph.add_edge(START, 'our_agent')



graph.add_conditional_edges(
    'our_agent',
    decide_next_step,
    {
        'continue': 'tools',
        'end':END
    }
)
app = graph.compile()

# ⬇파일 맨 아래에 이 실행 코드를 추가하세요!
if __name__ == "__main__":
    state = {"messages": []}
    
    print("=== 휴먼 인 더 루프 문서 작성 에이전트 시작 ===")
    
    # 유저가 강제로 종료하거나 완수할 때까지 무한 루프를 돕니다.
    while True:
        # stream_mode="values"로 현재 단계를 한 번 실행하고 최신 상태를 받아옵니다.
        for output in app.stream(state, stream_mode="values"):
            state = output # 대화 기록(state)을 계속 업데이트 누적합니다.
            
        # decide_next_step이 END를 가리켜서 그래프가 끝났는지 확인합니다.
        # 최신 상태의 메시지 중 마지막 툴 메시지에 'saved'가 들어있다면 루프를 종료합니다.
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "content") and "saved" in messages[-1].content.lower():
            print("\n 저장이 완료되어 시스템을 안전하게 종료합니다.")
            break