#Boilerplate code for future features that could be added

import autogen
from bs4 import BeautifulSoup

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

llm_config_research_web = {
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
        }
    ],
    "config_list": config_list   
}


web_outbound_researcher = autogen.AssistantAgent(
    name="web outbound researcher",
    system_message="Research the website of a potential lead and generate a detailed report; Add TERMINATE to the end of the research report;",
    llm_config=llm_config_research_web,
)

