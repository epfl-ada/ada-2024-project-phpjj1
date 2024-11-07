# Use the TMDB API to get movie adaptations based on books

import asyncio
import os
from dotenv import load_dotenv
import aiohttp
import json

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_READ_ACCESS_TOKEN = os.getenv("TMDB_READ_ACCESS_TOKEN")

with open("../data/booksmovies_list.txt", "r") as f:
    adaptations = f.read().splitlines()

async def make_request(session, page: int):
    url = f"https://api.themoviedb.org/3/keyword/818/movies?include_adult=false&language=en-US&page={page}"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_READ_ACCESS_TOKEN}"
    }

    async with session.get(url, headers=headers) as response:
        data = await response.json()
        
        print(f"Fetched page {page}/{data['total_pages']}: +{len(data['results'])} out of {data['total_results']} results")
        return {
            "results": data["results"],
            "total_pages": data["total_pages"], 
            "total_results": data["total_results"],
            "id": data["id"],
            "page": data["page"],
        }

async def main():
    async with aiohttp.ClientSession() as session:
        first_page = await make_request(session, 1)
        total_pages = first_page["total_pages"]
        
        all_results = [first_page]
        batch_size = 10
        
        for start_page in range(2, total_pages + 1, batch_size):
            end_page = min(start_page + batch_size, total_pages + 1)
            tasks = [make_request(session, page) for page in range(start_page, end_page)]
            batch_results = await asyncio.gather(*tasks) # wait every 10 pages
            all_results.extend(batch_results)
            
            # Small delay between batches to avoid rate limits
            await asyncio.sleep(0.5)
        
        with open("../data/tmdb_data.json", "w") as f:
            json.dump(all_results, f, indent=2)
            print(f"Saved {len(all_results)} pages to tmdb_data.json")

if __name__ == "__main__":
    asyncio.run(main())
