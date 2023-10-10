website_scraper_prompt = '''

Act as an expert outbound researcher.
Write a detailed summary of the following business based on the available data. I'm looking for information such as:
- Type of business with specific categorization
- Relevant clients
- Service Offered
- Number of people in the team
- Values of the business

"{text}"

SUMMARY:
'''

linkedin_scraper_prompt = '''

Act as an expert outbound researcher.
Write a detailed summary of the following person based on the Linkedin data. I'm lookinfor g information such as:
- Name of the companies that he/she worked with and roles
- Past results from clients
- Successfully transitioning from working in a job to starting their own business
- College education
- City where they are

"{text}"

SUMMARY:
'''