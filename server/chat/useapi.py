from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import json
import requests
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import LLMChain
from server.chat.utils import wrap_done
import asyncio
import aiohttp

stockApiBase="http://120.238.224.24:18080/json/aiquery/"


def tool_wrapper(tool):
    async def tool_(query):        
        query = json.loads(query)["query"]
        return tool(query)
    return tool_

def parseInput(jsonquery):
    return json.loads(jsonquery)["query"] 

def echoformat(data):
    return data;

def stockformat(data):
    pool = json.loads(data)
    result = "|股票名称|入选日期|买入价格|" + "\n|---|---|---|\n"
    for stock in pool:
        result += "|{stockName}|{chosenDate}|{chosenPrice}|\n".format(stockName=stock["stockName"], chosenDate=stock["chosenDate"], chosenPrice=stock["chosenPrice"])      
    return result

class donggaoapi:
    def __init__(self, wupfunc, outformater):
        self._wupfunc=wupfunc
        self._outputformat=outformater
        
    def run(self, query):
        url = stockApiBase + "/" + self._wupfunc
        data = {
            "stReq":{
                "text":parseInput(query)
            }
        }
        postjson = json.dumps(data)
        response = requests.post(url, data=postjson)
        if response.status_code == 200:
            rsp = json.loads(response.text)            
            return self._outputformat(rsp["stRsp"]["aitext"])
        return "未能查询到数据"
    
    async def arun(self, query):
        url = stockApiBase + "/" + self._wupfunc
        data = {
            "stReq":{
                "text":parseInput(query)
            }
        }
        postjson = json.dumps(data)
        response = requests.post(url, data=postjson)
        if response.status_code == 200:
            rsp = json.loads(response.text)            
            return self._outputformat(rsp["stRsp"]["aitext"])
        return "未能查询到数据"
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(url, data=postjson) as response:
        #         if response.status == 200:
        #             text = await response.text()
        #             rsp = json.loads(text)
        #             return self._outputformat(rsp["stRsp"]["aitext"])
        #         return ""        
 

lc_tools = [
    Tool(
        name='quote api', 
        func=donggaoapi('aiMarketInfo', echoformat).run, 
        coroutine=donggaoapi('aiMarketInfo', echoformat).arun, 
        description='用来查询股价信息，指数点位',
        return_direct=True,
    ),
    Tool(
        name='analyze api', 
        func=donggaoapi('aiOperateInfo', echoformat).run, 
        coroutine=donggaoapi('aiOperateInfo', echoformat).arun, 
        description='分析股价走势，分析公司经营情况，是否适合买入或者卖出',
        return_direct=True,
    ),
    Tool(
        name='stockPool', 
        func=donggaoapi('aiPoolInfo', stockformat).run, 
        coroutine=donggaoapi('aiPoolInfo', stockformat).arun, 
        description='分析股价走势，分析公司经营情况，是否适合买入或者卖出',
        return_direct=True,
    ),   
]

prefix="【指令】这是一个证券领域的查询, 请使用工具查询实时数据, 如果不能使用工具, 则直接回答‘不知道’, 请勿自己编造回答。api参数使用json格式, 参数名为query,值尽量使用是股票的中文名称"
suffix = """开始! 请记得使用中文回答. 
 
Question: {input}
{agent_scratchpad}"""


def can_choose_api(query):
    docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(lc_tools)]
    vector_store = FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name='moka-ai/m3e-base'), normalize_L2=True)
    docs = vector_store.similarity_search_with_score(query, k=5, score_threshold=0.5)
    if(len(docs) > 0):
        return True;
    else:
        return False;
    
def callagent(model, query)->str:
    prompt = ZeroShotAgent.create_prompt(
        lc_tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=model, prompt=prompt, return_final_only=True)
    tool_names = [tool.name for tool in lc_tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=lc_tools, verbose=True)
        
    return agent_executor.run(query)

async def callagentAsync(model, query):
    prompt = ZeroShotAgent.create_prompt(
        lc_tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=model, prompt=prompt, return_final_only=True)
    tool_names = [tool.name for tool in lc_tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=lc_tools, verbose=True)
        
    return await agent_executor.arun(query)
    # res = await agent_executor.arun(query)    
    # return res
    # await callback.on_llm_new_token("Observation:" )
    # await callback.on_llm_new_token(res)
    
    # return asyncio.create_task(wrap_done(
    #             agent_executor.arun(query),
    #             callback.done),
    #         )