import requests
import json
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from utils import prompts
import openai
from dotenv import load_dotenv
import os


def scrape_website(url:str):

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    data = {
        "url": url
    }

    data_json = json.dumps(data)
    response = requests.post("https://chrome.browserless.io/content?token=2db344e9-a08a-4179-8f48-195a2f7ea6ee", headers=headers, data=data_json)

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

def scrape_linkedIn(linkedin_url:str):
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
    
    return new_data['response']

