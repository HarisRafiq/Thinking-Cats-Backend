from typing import List
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import traceback

def web_search(queries: List[str]) -> str:
    """
    Performs web searches using DuckDuckGo for a list of queries.
    
    Args:
        queries: A list of search queries.
        
    Returns:
        A string containing the search results for each query.
    """
    import json
    results_data = []
    
    try:
        with DDGS() as ddgs:
            for query in queries:
                print(f"[Web Tools] Searching for: {query}")
                # Try with max_results=5 for better coverage
                results = list(ddgs.text(query, max_results=5))
                
                query_results = []
                if results:
                    for res in results:
                        query_results.append({
                            "title": res.get('title', ''),
                            "snippet": res.get('body', ''),
                            "url": res.get('href', '')
                        })
                
                results_data.append({
                    "query": query,
                    "results": query_results
                })
                
    except Exception as e:
        error_msg = f"Error performing web search: {str(e)}"
        print(f"[Web Tools] {error_msg}")
        traceback.print_exc()
        return json.dumps({"error": error_msg})

    return json.dumps(results_data, indent=2)

def web_page_reader(urls: List[str]) -> str:
    """
    Reads the content of web pages from a list of URLs.
    
    Args:
        urls: A list of URLs to read.
        
    Returns:
        A string containing the content of each page.
    """
    contents = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for url in urls:
        print(f"[Web Tools] Reading page: {url}")
        try:
            # Simple validation to ensure it's a URL
            if not url.startswith('http'):
                url = 'https://' + url
                
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading/trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long (to avoid context limit issues)
            if len(text) > 10000:
                text = text[:10000] + "... [Truncated]"
                
            contents.append(f"Content from {url}:\n{text}\n")
            
        except Exception as e:
            error_msg = f"Error reading {url}: {str(e)}"
            print(f"[Web Tools] {error_msg}")
            contents.append(error_msg)
            
    return "\n".join(contents)
