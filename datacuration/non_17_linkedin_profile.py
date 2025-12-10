import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict

def get_linkedin_profile(url: str, headers: dict) -> str:
    """
    Get LinkedIn profile page content
    
    Args:
        url: LinkedIn profile URL
        headers: Request headers including cookies and user agent
        
    Returns:
        HTML content of the page
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching profile: {e}")
        return None

def parse_experience_section(soup: BeautifulSoup) -> List[Dict]:
    """
    Parse LinkedIn experience section from BeautifulSoup object
    
    Args:
        soup: BeautifulSoup object of the profile page
        
    Returns:
        List of dictionaries containing job information
    """
    experiences = []
    
    # Find the experience section
    experience_section = soup.find('ul', {'class': 'fVNaDlLcOXJpcMpVsQErwUWBBVZnSDhImU'})
    if not experience_section:
        print("Experience section not found")
        return experiences
    
    
    experience_items = experience_section.find_all('li', class_='artdeco-list__item')
    
    for item in experience_items:
        experience = {}
        
        # Extract job title
        title_elem = item.find('div', class_='mr1 t-bold')
        if title_elem:
            experience['title'] = title_elem.get_text(strip=True)
            
        # Extract company name and employment type
        company_elem = item.find('span', class_='t-14 t-normal')
        if company_elem:
            company_text = company_elem.get_text(strip=True)
            company_parts = company_text.split('·')
            experience['company'] = company_parts[0].strip()
            if len(company_parts) > 1:
                experience['employment_type'] = company_parts[1].strip()
                
        # Extract duration
        duration_elem = item.find('span', class_='pvs-entity__caption-wrapper')
        if duration_elem:
            experience['duration'] = duration_elem.get_text(strip=True)
            
        # Extract location
        location_elem = item.find_all('span', class_='t-14 t-normal t-black--light')
        if location_elem and len(location_elem) > 1:
            experience['location'] = location_elem[-1].get_text(strip=True)
            
        # Extract description
        desc_elem = item.find('div', class_='inline-show-more-text--is-collapsed-with-line-clamp')
        if desc_elem:
            experience['description'] = desc_elem.get_text(strip=True)
            
        experiences.append(experience)
    
    return experiences

def main():
    # LinkedIn profile URL
    url = "https://www.linkedin.com/in/joonyoungyi/details/experience/"
    
    # You'll need to provide your own headers
    headers = {
        'User-Agent': 'Your User Agent String',
        'Cookie': 'Your LinkedIn cookies',  # Required for authentication
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    import pdb ; pdb.set_trace()
    
    # Get the page content
    html_content = get_linkedin_profile(url, headers)
    
    if html_content:
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract experience information
        experiences = parse_experience_section(soup)
        
        # Print results
        for i, exp in enumerate(experiences, 1):
            print(f"\nExperience {i}:")
            for key, value in exp.items():
                print(f"{key}: {value}")
    
if __name__ == "__main__":
    main()