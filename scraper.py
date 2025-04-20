import requests
from bs4 import BeautifulSoup
import random
from datetime import datetime
import time

class AgriculturalScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.sources = {
            'articles': [
                'https://krishijagran.com/',
                'https://www.agrifarming.in/',
                'https://plantvillage.psu.edu/'
            ],
            'remedies': [
                'https://agridoctor.in/',
            ],
            'chemical_treatments': [
                'https://www.cabidigitallibrary.org/product/qi',
                'https://icar.org.in/'
            ]
        }
        # Add rate limiting parameters
        self.min_delay = 2  # Minimum delay between requests in seconds
        self.max_delay = 5  # Maximum delay between requests in seconds
        self.last_request_time = 0

    def _polite_request(self, url):
        """Make a polite request with delay between requests"""
        # Calculate time since last request
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # If we haven't waited long enough, wait
        if time_since_last_request < self.min_delay:
            delay = random.uniform(self.min_delay, self.max_delay)
            time.sleep(delay)
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            self.last_request_time = time.time()
            return response
        except Exception as e:
            print(f"Error making request to {url}: {str(e)}")
            return None

    def get_random_source(self, category):
        return random.choice(self.sources[category])

    def scrape_articles(self, disease_name, limit=3):
        articles = []
        try:
            url = self.get_random_source('articles')
            response = self._polite_request(url)
            if not response:
                return articles
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This is a simplified example - actual implementation would need to be adjusted
            # based on the specific website structure
            article_elements = soup.find_all('article', class_='post')[:limit]
            
            for article in article_elements:
                title = article.find('h2').text.strip()
                link = article.find('a')['href']
                description = article.find('p', class_='excerpt').text.strip()
                
                articles.append({
                    'title': title,
                    'link': link,
                    'description': description,
                    'source': url,
                    'date': datetime.now().strftime('%Y-%m-%d')
                })
                
                # Add a small delay between processing articles
                time.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            print(f"Error scraping articles: {str(e)}")
        
        return articles

    def scrape_remedies(self, disease_name, limit=3):
        remedies = []
        try:
            url = self.get_random_source('remedies')
            response = self._polite_request(url)
            if not response:
                return remedies
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Simplified example - adjust based on actual website structure
            remedy_elements = soup.find_all('div', class_='remedy')[:limit]
            
            for remedy in remedy_elements:
                title = remedy.find('h3').text.strip()
                description = remedy.find('p', class_='description').text.strip()
                steps = [step.text.strip() for step in remedy.find_all('li')]
                
                remedies.append({
                    'title': title,
                    'description': description,
                    'steps': steps,
                    'source': url
                })
                
                # Add a small delay between processing remedies
                time.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            print(f"Error scraping remedies: {str(e)}")
        
        return remedies

    def scrape_chemical_treatments(self, disease_name, limit=3):
        treatments = []
        try:
            url = self.get_random_source('chemical_treatments')
            response = self._polite_request(url)
            if not response:
                return treatments
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Simplified example - adjust based on actual website structure
            treatment_elements = soup.find_all('div', class_='treatment')[:limit]
            
            for treatment in treatment_elements:
                name = treatment.find('h3').text.strip()
                description = treatment.find('p', class_='description').text.strip()
                dosage = treatment.find('div', class_='dosage').text.strip()
                precautions = [p.text.strip() for p in treatment.find_all('li', class_='precaution')]
                
                treatments.append({
                    'name': name,
                    'description': description,
                    'dosage': dosage,
                    'precautions': precautions,
                    'source': url
                })
                
                # Add a small delay between processing treatments
                time.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            print(f"Error scraping chemical treatments: {str(e)}")
        
        return treatments

    def get_all_information(self, disease_name):
        """Fetch all types of information for a given disease"""
        # Add a delay before starting a new batch of requests
        time.sleep(random.uniform(1, 3))
        
        return {
            'articles': self.scrape_articles(disease_name),
            'remedies': self.scrape_remedies(disease_name),
            'chemical_treatments': self.scrape_chemical_treatments(disease_name)
        }

# Example usage:
if __name__ == "__main__":
    scraper = AgriculturalScraper()
    info = scraper.get_all_information("Apple Scab")
    print("Articles:", len(info['articles']))
    print("Remedies:", len(info['remedies']))
    print("Chemical Treatments:", len(info['chemical_treatments'])) 