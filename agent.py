from turtle import mode
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage
from langchain.schema.messages import SystemMessage
from pathlib import Path
import base64
from typing import Dict, TypedDict, List
from typing import Optional
from pydantic import BaseModel, Field
load_dotenv()
from langgraph.graph import StateGraph, START, END

model = ChatOpenAI(model="gpt-4o-mini")
tavily_tool = TavilySearch(max_results = 5)
model_with_tavily = model.bind_tools([tavily_tool])
agent = create_react_agent(model, [tavily_tool])
search_platforms = ["AliExpress", "OLX-Uzbekistan"]

class AgentState(TypedDict):
    product_name: str
    aliexpress_result: str
    olx_result: str
    final_response: str
    image_path: str
graph = StateGraph(AgentState)


def identify_object(state: AgentState)->AgentState:
    def encode_image(image_path):
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")
    image_path = state["image_path"]
    image_encoded = encode_image(image_path)
    messages = [SystemMessage(content="You are an expert in identifying thing/object in the image given to you. only return object name.if brand name or anything that helps to specify object, also mention them. for example: instead of saying 'phone', you can respond 'iphone 17 pro max' if it is so.  don't add anything extra/explanatory/additional text"), 
                HumanMessage(content=[{"type": "text", "text": "identify a thing/object in the image given"}, 
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_encoded}"}}])]
    result = model.invoke(messages)
    response = result.content
    state["product_name"] = response
    return state

def search_internet(state: AgentState)->AgentState:
    Aliexpress = search_platforms[0]
    Olx = search_platforms[1]
    product_name = state["product_name"]
    ali_agent_response = agent.invoke({"messages": [{"role": "user", "content": f"use your search tool to search and get up to date, real info. give in this format: - Product name: str - Price: $float -Brand name: str(optional).   what are the prices of [{product_name}] in {Aliexpress}."}]})
    ali_agent_response = ali_agent_response["messages"][-1].content
    state["aliexpress_result"] = ali_agent_response

    olx_agent_response = agent.invoke({"messages": [{"role": "user", "content": f"use your search tool to search and get up to date, real info. give in this format: - Product name: str - Price: $float -Brand name: str(optional). make sure your search results exactly match with product. for example, if it is samsung remote control, but if your search results give samsung TV, it is wrong. make sure it matches. what are the prices of [{product_name}] in {Olx}."}]})
    olx_agent_response = olx_agent_response["messages"][-1].content
    state["olx_result"] = olx_agent_response
    return state

def finalize_response(state: AgentState)->AgentState:
    olx = state["olx_result"]
    ali = state['aliexpress_result']
    final = model.invoke(f"Here are results from 2 platforms. Merge them and display as plain text without any markdown formatting, asterisks, or special characters. but seperate two platform results.  Just return clean readable text. your response is displayed in frontend. so, don't add any extra/explanatory/additional text.\n\nAliExpress: {ali}\n\nOLX: {olx}")
    response = final.content
    state["final_response"] = response
    return state

graph.add_node("identify", identify_object)
graph.add_node("search", search_internet)
graph.add_node("finalize", finalize_response)
graph.add_edge(START, "identify")
graph.add_edge("identify", "search")
graph.add_edge("search", "finalize")
graph.add_edge("finalize", END)

app = graph.compile()






    



    









    
    