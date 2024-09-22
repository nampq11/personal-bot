import random
import os
import re
import json
import pandas as pd
import threading
import requests
from bs4 import BeautifulSoup
from loguru import logger
from os.path import dirname, join, exists
from typing import Optional
from time import sleep
from tqdm import tqdm
from pathlib import Path
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

class URLModel(BaseModel):
    url_question: str
    url_answer: str
    question: Optional[str] = None
    answer: Optional[str] = None

DATA_LOCATION = os.getenv('DATA_PATH')

all_data = []

BASE_URL = 'https://api-produce.isofhcare.com/consultation/v2/questions'

def get_urls():
    categories = {
        'num_ids': 48651,
    }
    for id in tqdm(range(1, categories['num_ids'] + 1), desc='getting urls'):
        yield URLModel(
                url_question=f'{BASE_URL}/{id}/',
                url_answer=f'{BASE_URL}/{id}/comments/',
            )

def save_json(data, filename):
    with open(os.path.join(DATA_LOCATION, filename), 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def scrape(conversation: URLModel):
    global all_data
    try:
        question = requests.get(conversation.url_question)
        answer = requests.get(conversation.url_answer)
        if question.status_code == 200 and answer.status_code == 200:
            logger.info(f'getting urls: {conversation.url_answer}')
            data = {
                'question': question.json()['content'],
                'answer': answer.json()['content'],
            }
            logger.info(f"question + answer from id {question.json()['id']} is: {data}")
            all_data.append(data)
    except Exception as e:
        logger.error(e)

if __name__ == '__main__':
    page_url = get_urls()
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(scrape, url) for url in page_url]
        for future in tqdm(as_completed(futures), total=len(futures), desc='scraping data'):
            future.result()

    save_json(all_data, 'data.json')
    logger.info(all_data)