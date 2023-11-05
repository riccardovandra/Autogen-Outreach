import requests
import json
import autogen
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from utils import prompts
import openai
from dotenv import load_dotenv
import os

#get openai API keys
load_dotenv()
config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize(content, type):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    if type == 'linkedin':
        map_prompt = prompts.linkedin_scraper_prompt
    elif type == 'website':
        map_prompt = prompts.website_scraper_prompt

    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs,)

    return output

def scrape_linkedin(linkedin_url:str):
    json_cache = 'json_cache.json'
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    api_key = os.getenv('PROXYCURL_API')
    header_dic = {'Authorization': 'Bearer ' + api_key}
    params = {
        'linkedin_profile_url': linkedin_url,
        'use_cache': 'if-present',
    }

    ## Try to fetch data from local cache
    try:
        with open(json_cache, 'r') as f:
            cached_data = json.load(f)
            for entry in cached_data:
                if entry['linkedin_url'] == linkedin_url:
                    print('Fetched data from Local Cache')
                    return entry['response']
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f'No local cache found...({e})')
        cached_data = []
    
    # If data not found in cache, make an API call
    print('Fetching new json data... (updating local cache)')
    response = requests.get(api_endpoint,
                        params=params,
                        headers=header_dic)
    
    new_data = {
        'linkedin_url': linkedin_url,
        'response': response.json()
    }
    cached_data.append(new_data)
    
    # Update the local cache with new data
    with open(json_cache, 'w') as f:
        json.dump(cached_data, f,  indent=4)
    
    output = summarize(new_data['response'],'linkedin')

    return output

# Create the research function

def research(lead_data:dict):
    llm_config_research_li = {
        "functions" : [
            {
                "name": "scrape_linkedin",
                "description": "scrape the website and look for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "linkedin_url": {
                            "type": "string",
                            "description": "The Linkedin URL to scrape",
                        }
                    },
                    "required": ["linkedin_url"],
                },
            }
        ],
        "config_list": config_list   
    }

    outbound_researcher = autogen.AssistantAgent(
        name="Outbound_researcher",
        system_message="Research the LinkedIn Profile of a potential lead and generate a detailed report; Add TERMINATE to the end of the research report;",
        llm_config=llm_config_research_li,
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
        max_consecutive_auto_reply = 3,
        default_auto_reply = 'Please continue with the task',
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        function_map={
            "scrape_linkedin": scrape_linkedin
            }
    )

    user_proxy.initiate_chat(outbound_researcher, message=f"Research this lead's website and LinkedIn Profile {str(lead_data)}")

    user_proxy.stop_reply_at_receive(outbound_researcher)
    user_proxy.send(
        "Give me the research report that just generated again, return ONLY the report", outbound_researcher)

    # return the last message the expert received
    return user_proxy.last_message()["content"]


# Create the outreach creation function

def create_outreach_msg(research_material, lead:dict):
    outbound_strategist = autogen.AssistantAgent(
        name="outbound_strategist",
        system_message="You are a senior outbound strategist responsable for analyzing research material and coming up with the best cold email structure with relevant personalization points",
        llm_config={"config_list": config_list},
    )

    outbound_copywriter = autogen.AssistantAgent(
        name="outbound_copywriter",
        system_message="You are a professional AI copywriter who is writing cold emails for leads. You will write a short cold email based on the structured provided by the outbound strategist, and feedback from the reviewer; After 2 rounds of content iteration, add TERMINATE to the end of the message",
        llm_config={"config_list": config_list},
    )

    reviewer = autogen.AssistantAgent(
        name="reviewer",
        system_message="You are a world class cold email critic, you will review & critic the cold email and provide feedback to writer.After 2 rounds of content iteration, add TERMINATE to the end of the message",
        llm_config={"config_list": config_list},
    )

    user_proxy = autogen.UserProxyAgent(
        name="admin",
        system_message="A human admin. Interact with outbound strategist to discuss the structure. Actual writing needs to be approved by this admin.",
        code_execution_config=False,
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy, outbound_strategist, outbound_copywriter, reviewer],
        messages=[],
        max_round=20)
    manager = autogen.GroupChatManager(groupchat=groupchat)

    user_proxy.initiate_chat(
        manager, message=f"Write a personalized cold email to {lead}, here are the material: {research_material}")

    user_proxy.stop_reply_at_receive(manager)
    user_proxy.send(
        "Give me the cold email that just generated again, return ONLY the cold email, and add TERMINATE in the end of the message", manager)

    # return the last message the expert received
    return user_proxy.last_message()["content"]


llm_config_outbound_writing_assistant = {
    "functions": [
        {
            "name": "research",
            "description": "research about a given lead, return the research material in report format",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "lead_data": {
                            "type": "object",
                            "description": "The information about a lead",
                        }
                    },
                "required": ["lead_data"],
            },
        },
        {
            "name": "create_outreach_msg",
            "description": "Write an outreach message based on the given research material & lead information",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "research_material": {
                            "type": "string",
                            "description": "research material of a given topic, including reference links when available",
                        },
                        "lead": {
                            "type": "object",
                            "description": "A dictionary containing lead data",
                        }
                    },
                "required": ["research_material", "lead"],
            },
        },
    ],
    "config_list": config_list}


outbound_writing_assistant = autogen.AssistantAgent(
    name="writing_assistant",
    system_message="You are an outbound assistant, you can use research function to collect information from a lead, and then use create_outreach_msg function to write a personalized outreach message; Reply TERMINATE when your task is done",
    llm_config=llm_config_outbound_writing_assistant,
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    human_input_mode="TERMINATE",
    function_map={
        "create_outreach_msg": create_outreach_msg,
        "research": research,
    }
)

#Generate Lead Data
lead_data = {
    'First Name': 'Mulenga',
    'Company Name': 'Growthcurve',
    'Website URL' : 'http://growthcurve.co',
    'LinkedIn URL': 'https://www.linkedin.com/in/mulengaagley'
}

user_proxy.initiate_chat(
    outbound_writing_assistant, message=f"create an effective outreach message for the following lead {str(lead_data)}")

