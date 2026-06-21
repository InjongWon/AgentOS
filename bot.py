# define structure a list of Human Message objects
# initialize gpt-4o using langchain's chat open ai 
# sending and handling different types of messages
# building and compiling the graph of the agent
#Main goal: how llm interacts 
from typing import TypedDict,List, Annotated
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, START, END # langgraph is built on top of langchain
from dotenv import load_dotenv # used to store secret stuff like API
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage], add_messages]
    
llm = ChatOpenAI(model = "gpt-4o", temperature = 0.5)


def process(state:AgentState)->AgentState:
    "process the message"
    res = llm.invoke(state['messages'])
    print(f"\nAO:{res.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node('process', process)
graph.add_edge(START,'process')
graph.add_edge('process', END)
agent = graph.compile()

state = {'messages': []}


while True:
    user_input = input("enter: ")
    if user_input.lower() == "exit":
        break
    
    state = agent.invoke({'messages': [HumanMessage(content=user_input)]})
    
    print(f"\nAI:", state['messages'][-1].content)
    
# from IPython.display import Image, display

# 코드 로직
# try:
#     # 1.그래프 이미지 바이트 데이터를 가져옵니다.
#     graph_image_bytes = agent.get_graph().draw_mermaid_png()
    
#     # 2.로컬 저장.
#     with open("graph.png", "wb") as f:
#         f.write(graph_image_bytes)
#     print("\n그래프 시각화 완료! 프로젝트 폴더 내 'graph.png' 확인.")

# except Exception as e:
#     print(f"\n 그래프 시각화 실패: {e}")
#     print("팁: 패키지 누락일 수 있으니 터미널에 'pip install requests.")