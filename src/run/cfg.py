import os
import sys
from pydantic import BaseModel, ConfigDict, field_validator
from loguru import logger
from typing import Literal, List

from src.run.args import RunInputArgs
from src.run.utils import pprint_pydantic_model, substitute_punctuation

response_curated_eval_dataset_fp = "data/031_rerun/response_curated_eval_dataset.json"
response_synthetic_eval_dataset_fp = (
    "/home/nampq/Desktop/projects/personal-bot/notebooks/data/01_create_retrieval_dataset/retrieval_synthetic_eval_dataset.json"
)
retrieval_synthetic_eval_dataset_fp = (
    "/home/nampq/Desktop/projects/personal-bot/notebooks/data/01_create_retrieval_dataset/retrieval_synthetic_eval_dataset.json"
)
# storage_context_persist_dp = "/home/nampq/Desktop/projects/personal-bot/data/031_rerun/storage_context"
storage_context_persist_dp = "/home/nampq/Desktop/projects/personal-bot/data/001_rerun/storage_context"
# db_collection = "review_rec_bot_031_rerun"
db_collection = "healthcare_bot_032_rerun"
db_collection_fp = "/home/nampq/Desktop/projects/personal-bot/data/031_rerun/chroma_db"

class LLMConfig(BaseModel):
    llm_provider: Literal["openai", "togetherai", "ollama", "gemini"] = "gemini"
    llm_model_name: str = "models/gemini-1.5-flash"
    # llm_model_name: str = "qwen2.5:7b"
    # llm_model_name: str = "llama3.1:8b"
    embedding_provider: Literal["openai", "togetherai", "ollama", "huggingface", "optimum"] = (
        "optimum"
    )
    embedding_model_name: str = "/home/nampq/Desktop/projects/personal-bot/data/BookingCare/multilingual-e5-base-v2-onnx-quantized"
    embedding_model_dim: int = None

    ollama__host: str = "https://5cb2-34-74-225-158.ngrok-free.app/"
    ollama__port: int = 11434


class RetrievalConfig(BaseModel):
    retrieval_top_k: int = 50
    retrieval_dense_top_k: int = 50
    retrieval_sparse_top_k: int = 50
    retrieval_simiarity_cutoff: int = (
        None
    )
    rerank_top_k: int = 10
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"

class EvalConfig(BaseModel):
    retrieval_num_sample_nodes: int = 30
    retrieval_eval_llm_model: str = "llama3.1:8b"
    retrieval_eval_llm_model_config: dict = {"temperature": 0.3}
    retrieval_eval_num_questions_per_chunk: int = 1
    retrieval_metrics: List[str] = [
        "hit_rate",
        "mrr",
        "precision",
        "recall",
        "ap",
        "ndcg",
    ]
    retrieval_eval_dataset_fp: str = retrieval_synthetic_eval_dataset_fp

#     response_question_gen_query: str = """
# you are a helpul assistant.

# your task is to generate {num_questions_per_chunk} questions based on only given context, not prior information.
# The questions are aim to find healthcare service to go to, for example: doctor, clinic, clinicPlace, disease, ...

# <EXAMPLE>
# Input context: Để điều trị nấm da đầu, bạn có thể áp dụng các phương pháp sau:\n1. Sử dụng thuốc điều trị theo đường uống: Thông thường, các trường hợp nấm da đầu cần phải sử dụng thuốc uống kháng nấm để điều trị. Các loại thuốc thường được sử dụng bao gồm Griseofulvin, Terbinafine, Itraconazole, Fluconazole. Thời gian sử dụng thuốc tùy thuộc vào từng loại và được chỉ định bởi bác sĩ.\n2. Đảm bảo vệ sinh da đầu: Vệ sinh da đầu thường xuyên và sạch sẽ là rất quan trọng trong việc điều trị nấm da đầu. Hãy gội đầu hàng ngày bằng shampoo chuyên dụng chống nấm và sử dụng lược, khăn tay riêng để tránh lây nhiễm.\n3. Tránh để tóc ướt, ẩm khi đi ngủ: Nấm da đầu thích môi trường ẩm ướt, do đó hãy đảm bảo tóc khô ráo trước khi đi ngủ để không tạo điều kiện cho nấm phát triển.
# Output questions: Điều trị nấm da đầu như thế nào?

# Some example of good generated questions: 
# - bệnh viện chợ rẫy ở đâu?
# - Rối loạn thần kinh thực vật có nguy hiểm không?
# </EXAMPLE>

# IMPORTANT RULES:
# - The generated questions must be specific about the categories of businesses it's  looking for. A good generated question would have its typical answer being: Here are some options for you: Place A because..., Place B because...
# - Restrict the generated questions to the context information provided
# - Pay attention to the sentiment of the context review. If the review is bad then never return a question that ask for a good experience.
# - Do not mention anything about the context in the generated queries
# - The generated questions must be complete on its own. Do not assume the person receiving the question know anything about the person asking the question. for example never use "in my area" or "near me
# ."""
    response_question_gen_query: str = """
bạn là một trợ lý hữu ích.

nhiệm vụ của bạn là tạo ra {num_questions_per_chunk} câu hỏi chỉ dựa trên ngữ cảnh nhất định chứ không phải thông tin trước đó.
Các câu hỏi nhằm mục đích tìm kiếm thông tin về dịch vụ y tế, ví dụ: bệnh, bác sĩ, dịch vụ y tế, ...
<VÍ DỤ>
input context: Để giảm thời gian chờ đợi và nhận được hướng dẫn đi khám tại Bệnh viện Chợ Rẫy, người bệnh vui lòng:\nChọn chuyên khoa phù hợp cần đi khám\nChọn thời gian đặt khám\nĐặt hẹn online trước khi đến khám. \nGIỚI THIỆU\nNội Phổi, Bệnh viện Chợ Rẫy\nĐịa chỉ:\nKhu A Bệnh viện Chợ Rẫy - số 201B Nguyễn Chí Thanh, Phường 12, Quận 5, Hồ Chí Minh\nThời gian làm việc:\nBệnh viện làm việc, tiếp nhận khám bệnh cho bệnh nhân ngoại trú từ thứ 2 đến thứ 7 hàng tuần:\nThứ 2 đến thứ 6: từ 7h – 16h\nThứ 7: từ 7h – 11h\nLưu ý thông tin đặt lịch: \nVui lòng kiểm tra kĩ thông tin cá nhân (họ tên, giới tính, năm sinh, địa chỉ, BHYT) trước khi xác nhận đặt lịch khám để tránh sai sót và tiết kiệm thời gian (\nBệnh viện không chấp nhận với lịch sai thông tin) \nTrường hợp đặt sai thông tin, vui lòng đặt lại bằng số điện thoại khác.
ouput questions: địa chỉ của bệnh viện Chợ Rẫy là gì?

Một số ví dụ về các câu hỏi được tạo tốt: 
- Quy trình khám và làm hồ sơ sinh tại Bệnh viện Bạch Mai diễn ra như thế nào?
- Điều trị nấm da đầu như thế nào?
</VÍ DỤ>

QUY TẮC QUAN TRỌNG:
- Các câu hỏi được tạo ra phải cụ thể về cụ thể  về một dịch vụ y tế đang tìm kiếm. Một câu hỏi hay sẽ có câu trả lời điển hình là: Dưới đây là thông tin về yêu cầu của bạn: Thông tin về bác sĩ A,..., Thông tin về bệnh viện A
- Hạn chế các câu hỏi được tạo ra đối với thông tin ngữ cảnh được cung cấp
- Chú ý đến cảm xúc khi xem xét bối cảnh. 
- Không đề cập bất cứ điều gì về bối cảnh trong các truy vấn được tạo
- Các câu hỏi được tạo ra phải tự hoàn chỉnh. Đừng cho rằng người nhận câu hỏi biết bất cứ điều gì về người đặt câu hỏi. ví dụ: không bao giờ sử dụng "trong khu vực của tôi" hoặc "gần tôi"
"""
    response_synthetic_eval_dataset_fp: str = response_synthetic_eval_dataset_fp
    response_curated_eval_dataset_fp: str = response_curated_eval_dataset_fp
    response_eval_llm_model: str = "llama3.1:8b"
    response_eval_llm_model_config: dict = {"temperature": 0.3}
    response_synthetic_num_questions_per_chunk: int = 1
    response_num_sample_documents: int = 30

class RunConfig(BaseModel):
    args: RunInputArgs = None
    app_name: str = "review_rec_bot"
    storage_context_persist_dp: str = storage_context_persist_dp
    vector_db: Literal["chromadb", "qdrant"] = "qdrant"
    db_collection: str = db_collection
    db_collection_fp: str = db_collection_fp
    notebook_cache_dp: str = None

    data_fp: str = "../data/yelp_dataset/sample/sample_100_biz/denom_review.parquet"

    llm_cfg: LLMConfig = LLMConfig()

    retrieval_cfg: RetrievalConfig = RetrievalConfig()

    eval_cfg: EvalConfig = EvalConfig()

    batch_size: int = 1 # Prevent Out of GPU Memory

    def init(self, args: RunInputArgs):
        self.args = args

        if args.OBSERVABILITY: 
            logger.info(f"Starting Observability server with Phoenix...")
            import phoenix as px
            px.launch_app()
            import llama_index.core

            llama_index.core.set_global_handler("arize_phoenix")
        if args.DEBUG:
            logger.info(f"Enabling LlamaIndex DEBUG logging...")
            import logging

            logging.getLogger("llama_index").addHandler(
                logging.StreamHandler(stream=sys.stdout)
            )
            logging.getLogger("llama_index").setLevel(logging.DEBUG)
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            logger.warning(
                f"Enviroment variable MLFLOW_TRACKING_URI is not set. Setting args.LOG_TO_MLFLOW to false."
            )
            args.LOG_TO_MLFLOW = False
        
        if args.LOG_TO_MLFLOW:
            logger.info(
                f"Setting up MLflow experiment {args.EXPERIMENT_NAME} - run {args.RUN_NAME}..."
            )
            import mlflow

            mlflow.set_experiment(args.EXPERIMENT_NAME)
            mlflow.start_run(run_name=args.RUN_NAME, description=args.RUN_DESCRIPTION)
        
        self.notebook_cache_dp = f"data/{args.RUN_NAME}"
        logger.info(
            f"Notebook-generated artifacts are persisted at {self.notebook_cache_dp}"
        )
        os.makedirs(self.notebook_cache_dp, exist_ok=True)

        if args.RECREATE_INDEX:
            logger.info(
                f"ARGS.RECREATE_INDEX=True -> Overwriting db_collection and storage_context_persist_dp..."
            )
            collection_raw_name = f"{self.app_name}__{args.RUN_NAME}"
            self.storage_context_persist_dp = (
                f"{self.notebook_cache_dp}/storage_context"
            )
            self.db_collection_fp = f"{self.notebook_cache_dp}/{self.vector_db}"
            self.db_collection = substitute_punctuation(collection_raw_name)

        if args.TESTING:
            logger.info(
                f"TESTING=True -> Limiting the number of eval questions generated to 2..."
            )
            self.eval_cfg.retrieval_num_sample_nodes = 2
            self.eval_cfg.response_num_sample_documents = 2
    
    def setup_llm(self):
        # set up LLM
        llm_provider = self.llm_cfg.llm_provider
        llm_model_name = self.llm_cfg.llm_model_name
        
        if llm_provider == "ollama":
            import subprocess
            from llama_index.llms.ollama import Ollama

            ollama_host = self.llm_cfg.ollama__host
            ollama_port = self.llm_cfg.ollama__port

            # base_url = f"http://{ollama_host}:{ollama_port}"
            base_url = f"{ollama_host}"
            print(base_url)
            llm = Ollama(
                base_url=base_url,
                model=llm_model_name,
                request_timeout=60.0,
            )
            command = ["ping", "-c", "1", ollama_host]
            subprocess.run(command, capture_output=True, text=True)
        elif llm_provider == "openai":
            from llama_index.llms.openai import OpenAI
            llm = OpenAI(
                model=llm_model_name,
                temperature=0,
            )
        elif llm_provider == "togetherai":
            from llama_index.llms.together import TogetherLLM
            llm = TogetherLLM(
                model=llm_model_name,
            )
        elif llm_provider == "gemini":
            from llama_index.llms.gemini import Gemini
            llm = Gemini(
                model=llm_model_name,
            )

        # set up embedding model
        embedding_provider = self.llm_cfg.embedding_provider
        embedding_model_name = self.llm_cfg.embedding_model_name

        if embedding_provider == "huggingface":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            embed_model = HuggingFaceEmbedding(
                model_name=embedding_model_name,
                embed_batch_size=4,
                trust_remote_code=True,
                device="cpu",
            )
        elif embedding_provider == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed_model = OpenAIEmbedding()
        elif embedding_provider == "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding
            embed_model = OllamaEmbedding(
                model_name=embedding_model_name,
                base_url=base_url,
                ollama_additional_kwargs={'mirostat': 0},
            )
        elif embedding_provider == "optimum":
            from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
            embed_model = OptimumEmbedding(
                folder_name=embedding_model_name,
                device="cpu",
            )
        self.llm_cfg.embedding_model_dim = len(
            embed_model.get_text_embedding("sample text")
        )

        return llm, embed_model
    
    def __repr__(self):
        return pprint_pydantic_model(self)

    def __str__(self):
        return pprint_pydantic_model(self)