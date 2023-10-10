import requests
import json
from bs4 import BeautifulSoup
import cloudscraper
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

#Helper function

# def scrape(url:str):
#     if url.startswith('https://www.linkedin.com'):
#         scrape_linkedin(url)
#     else:
#         scrape_website(url)

def scrape_website(website_url:str):

    # Create a session
    
    scraper = cloudscraper.create_scraper()
    
    # Set headers
    headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.57"}
    
    # Send a GET request to the URL using the session
    response = scraper.get(website_url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)
        if len(text) > 8000:
            output = summarize(text,type='website')
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

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
    llm_config_research = {
        "functions" : [
            {
                "name": "scrape_website",
                "description": "scrape the website and look for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "website_url": {
                            "type": "string",
                            "description": "The Website URL to scrape",
                        }
                    },
                    "required": ["website_url"],
                },
            },
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
        name="outbound_researcher",
        system_message="Research about a given prospect, collect as many information as possible, and generate detailed report with every single information that could be used in our outreach efforts; Add TERMINATE to the end of the research report;",
        llm_config=llm_config_research,
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        function_map={
            "scrape_website": scrape_website,
            "scrape_linkedin": scrape_linkedin
            }
    )

    user_proxy.initiate_chat(outbound_researcher, message=f"Research this lead's website and LinkedIn Profile {str(lead_data)}")

    # set the receiver to be researcher, and get a summary of the research report
    user_proxy.stop_reply_at_receive(outbound_researcher)

    # return the last message the expert received
    return user_proxy.last_message()["content"]


#Generate Lead Data
lead_data = {
    'First Name': 'Mulenga',
    'Company Name': 'Growthcurve',
    'Website URL' : 'http://growthcurve.co',
    'LinkedIn URL': 'https://www.linkedin.com/in/mulengaagley'
}


#Run the research function
research(lead_data)