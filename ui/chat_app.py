import os
import sys

import chainlit as cl
import pandas as pd
import torch
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import (QueryFusionRetriever,
                                         VectorIndexRetriever)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.llms.gemini import Gemini
from llama_index.postprocessor.flag_embedding_reranker import \
    FlagEmbeddingReranker
from llama_index.retrievers.bm25 import BM25Retriever
from loguru import logger

# from ui.callback_handler import LlamaIndexCallbackHandler

sys.path.insert(0, "..")

from src.run.args import RunInputArgs
from src.run.cfg import RunConfig

load_dotenv()

USE_GPU = torch.cuda.is_available()

if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger.info(f"{torch.cuda.is_available()}")

ARGS = RunInputArgs(
    EXPERIMENT_NAME="Healthcare chatbot",
    RUN_NAME="03_rerun_healthcare",
    RUN_DESCRIPTION="""
# Objective
""",
    TESTING=False,
    LOG_TO_MLFLOW=False,
    OBSERVABILITY=True,
    RECREATE_INDEX=False,
    RECREATE_RESPONSE_EVAL_DATASET=False,
    DEBUG=False,
)

logger.info(f"{ARGS}")

cfg = RunConfig()

dir_prefix = "../notebooks"
cfg.storage_context_persist_dp = (
    "/home/nampq/Desktop/projects/personal-bot/data/001_rerun/storage_context"
)
cfg.db_collection = "healthcare_bot_032_rerun"
cfg.db_collection_fp = (
    "/home/nampq/Desktop/projects/personal-bot/data/031_rerun/chroma_db"
)
cfg.llm_cfg.embedding_model_name = "/home/nampq/Desktop/projects/personal-bot/data/BookingCare/multilingual-e5-base-v2-onnx-quantized"
cfg.data_fp = "../data/yelp_dataset/sample/sample_100_biz/denom_review.parquet"

cfg.init(ARGS)

logger.info(cfg)

llm, embed_model = cfg.setup_llm()

logger.info(cfg.llm_cfg.model_dump_json(indent=2))

Settings.embed_model = embed_model
Settings.llm = llm

if cfg.vector_db == "qdrant":
    from src.run.vector_db.qdrant import QdrantVectorDB as VectorDB

vector_db = VectorDB(cfg=cfg)
vector_store = vector_db.vector_store
db_collection_count = vector_db.doc_count
logger.info(f"{db_collection_count=}")

logger.info(f"Loading Storage Context from {cfg.storage_context_persist_dp}...")
docstore = SimpleDocumentStore.from_persist_dir(cfg.storage_context_persist_dp)
storage_context = StorageContext.from_defaults(
    docstore=docstore,
    vector_store=vector_store,
)
nodes = list(docstore.docs.values())

logger.info(f"[COLLECT] {len(nodes)=}")

logger.info(f"Confiuring Vector Retriever...")
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
)
vector_retriever = VectorIndexRetriever(
    index=index,
    vector_store_query_mode="mmr",
    similarity_top_k=cfg.retrieval_cfg.retrieval_dense_top_k,
)

logger.info(f"Configuring bm25 retriever...")
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=cfg.retrieval_cfg.retrieval_sparse_top_k,
)

logger.info(f"Configuring Query Fusion Retriever...")
query_gen_prompt = """
You are a helpful assistant that expands an input query into new strings that aim to increase the recall of an information retrieval system. The strings can be queries or paragraphs or sentences.
You should apply different techniques to create new strings. Here are some example techniques:
- Technique 1 - Optimize for full-text search: Rephrase the input query to contain only important keywords. Remove all stopwords and low information words. Example input query: "What are some places to enjoy cold brew coffee in Hanoi?" -> Expected output:  "cold brew coffee hanoi"
- Technique 2 - Optimize for similarity-based vector retrieval: Create a fake user review that should contain the answer for the question. Example input query: "What are some good Pho restaurants in Singapore?" -> Expected output query: "I found So Pho offerring a variety of choices to enjoy not Pho but some other Vietnamese dishes like bun cha. The price is reasonable."

Generate at least {num_queries} new strings by iterating over the technique in order. For example, your first generated string should always use technique 1, second technique 2. If run of of techniques then re-iterate from the start.

Return one string on each line, related to the input query.

Only return the strings. Never include the chosen technique.

Input Query: {query}\n
New strings:\n
"""


llm = Gemini(model_name=cfg.llm_cfg.llm_model_name, temperature=0.3)

logger.info(f"Setting up Post-Retriever Processor...")
node_postprocessors = []
if cfg.retrieval_cfg.retrieval_simiarity_cutoff is not None:
    node_postprocessors.append(
        SimilarityPostprocessor(
            similarity_cutoff=cfg.retrieval_cfg.retrieval_simiarity_cutoff
        )
    )

logger.info(f"Registering Query Engine as Tool...")

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    llm=llm,
    similarity_top_k=cfg.retrieval_cfg.retrieval_top_k,
    num_queries=2,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
    query_gen_prompt=query_gen_prompt,
)

healthcare = QueryEngineTool(
    query_engine=retriever,
    metadata=ToolMetadata(
        name="healthcare",
        description=(
            "useful for when you want to find healthcare related information"
            "based on user query."
            "for example, 'bệnh viện chợ rẫy ở đâu?'"
        ),
    ),
)


# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

tools = [healthcare, multiply_tool]


agent_system_prompt = """
Bạn là trợ lý ảo của BookingCare, hãy giúp tôi tìm kiếm thông tin.

khi người dùng hỏi thông tin về bệnh viện, bệnh, triệu chứng, phòng khám hãy luôn luôn sử dụng query_engine_tool. Nếu bạn không hiểu thông tin, làm ơn hỏi lại người dùng.

bạn phải trả lại nguồn cho người dùng để họ biết được thông tin lấy ra ở đâu.

Nếu có các nguồn trích dẫn được trả về từ các công cụ, hãy luôn trả lại chúng chính xác như câu trả lời của bạn cho người dùng.
Điều này có nghĩa là bạn phải tôn trọng vị trí của số trích dẫn (như [1], [2]) trong câu trả lời và ở cuối bên dưới phần Nguồn.
"""


@cl.on_chat_start
async def start():
    # agent = OpenAIAgent.from_tools(
    #     tools=tools,
    #     verbose=True,
    #     llm=llm,
    #     system_prompt=agent_system_prompt,
    #     # callback_manager=CallbackManager([LlamaIndexCallbackHandler()]),
    # )
    agent = ReActAgent.from_tools(
        llm=llm,
        tools=tools,
        verbose=True,
        system_prompt=agent_system_prompt,
        max_iterations=10,
    )

    cl.user_session.set("agent", agent)

    await cl.Message(
        author="bookingcare",
        content="Xin chào, tôi là trợ lý ảo của BookingCare, hãy giúp tôi tìm kiếm thông tin.",
    ).send()


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")

    msg = cl.Message(content="", author="bookingcare")

    res = await cl.make_async(agent.chat)(message.content)

    for token in res.response:
        await msg.stream_token(token)

    await msg.send()
