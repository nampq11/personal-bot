import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import pandas as pd
import translators as ts
from datasets import load_from_disk
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()

DATA_LOCATION = os.getenv("DATA_PATH")


logger.info("Loading data from {}".format(os.path.join(DATA_LOCATION, "part1")))

full_data = load_from_disk(os.path.join(DATA_LOCATION, "part1"))
translated_data = []


class EmbeddingData(BaseModel):
    sentence1: str
    sentence2: str


def get_datas(batch_size=256):
    # generate batch data from full_data
    embedding_data = []
    for data in full_data:
        embedding_data.append(
            EmbeddingData(sentence1=data["sentence1"], sentence2=data["sentence2"])
        )
        if len(embedding_data) == batch_size:
            yield embedding_data
            embedding_data = []


def translate(embedding_data: EmbeddingData) -> Dict[str, str]:
    try:
        data = {
            "sentence1": ts.translate_text(
                query_text=embedding_data.sentence1,
                translator="bing",
                from_language="auto",
                to_language="vi",
            ),
            "sentence2": ts.translate_text(
                query_text=embedding_data.sentence2,
                translator="bing",
                from_language="auto",
                to_language="vi",
            ),
        }
        return data
    except Exception as e:
        logger.error(e)


def save_json(data, filename):
    logger.info(f"Saving {len(data)} data to {filename}")
    with open(os.path.join(DATA_LOCATION, filename), "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def run(data):
    global translated_data
    try:
        logger.info(f"Translating {len(data)} data: {data}")
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(translate, text) for text in data]
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    translated_data.append(result)
        logger.info(translated_data)
        save_json(translated_data, "translated_data1.json")
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    english_data = get_datas()
    for batch in tqdm(english_data, total=len(full_data) // 256):
        logger.info(f"Translating {len(batch)} data")
        run(batch)
