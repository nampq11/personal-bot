from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.utils import is_chat_model

CUSTOM_TREE_SUMMARIZE_TMPL = (
    "Context information about various restaurants is provided below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Based on only the provided information, recommend multiple restaurants that match the user's preferences."
    "You should be rank the recommendations based on how relevant they are to user's query"
    "Provide a summary explanation of strengths of each option and compare them with each other based on different intentions.\n"
    "User Query: {query_str}"
    "Recommendations:"
)
CUSTOM_TREE_SUMMARIZE_PROMPT = PromptTemplate(
    CUSTOM_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
)

CUSTOM_TREE_SUMMARIZE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=CUSTOM_TREE_SUMMARIZE_PROMPT,
)