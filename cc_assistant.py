
import logging
import os
from typing import Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import re
import difflib
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DATA_FILE = "creditcards.xlsx"
FAISS_INDEX_DIR = "./faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/mistral-7b-instruct:free"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
SCORING_WEIGHTS = {
    "annual_fee": 0.3,
    "reward_rate": 0.3,
    "lounge_access": 0.2,
    "luxury_perks": 0.15,
    "welcome_bonus": 0.05,
}
ALLOWED_CARD_FIELDS = {
    "credit_card_name",
    "annual_fee",
    "cardmaven_review",
    "cardexpert_review",
    "best_tags",
}

class CreditCardAssistantError(Exception):
    """Base exception for CreditCardAssistant errors."""
    pass

class DataLoadingError(CreditCardAssistantError):
    """Raised when data loading fails."""
    pass

class VectorStoreError(CreditCardAssistantError):
    """Raised when vector store operations fail."""
    pass

class CreditCardAssistant:
    """Assistant for credit card queries using LangChain and vector search."""

    def __init__(self, data_path: str, faiss_index_dir: str) -> None:
        """Initialize the assistant with data and vector store paths.

        Args:
            data_path: Path to the credit card Excel file.
            faiss_index_dir: Directory for FAISS vector store.

        Raises:
            CreditCardAssistantError: If initialization fails.
        """
        load_dotenv()
        self.api_key = "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"
        if not self.api_key:
            raise CreditCardAssistantError("OPENROUTER_API_KEY not set.")

        self.data_path = Path(data_path)
        self.faiss_index_dir = Path(faiss_index_dir)
        self.df = self._load_data()
        self.vectorstore = self._setup_vectorstore()
        self.llm = self._setup_llm()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5})
        )
        self.agent = self._setup_agent()

    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess credit card data from Excel.

        Returns:
            DataFrame with cleaned column names and combined content.

        Raises:
            DataLoadingError: If file is missing, empty, or malformed.
        """
        try:
            if not self.data_path.exists():
                raise DataLoadingError(f"File not found: {self.data_path}")
            df = pd.read_excel(self.data_path)
            if df.empty:
                raise DataLoadingError("Excel file is empty.")
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            def clean_currency_column(df, column):
                df[column] = df[column].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False)
                df[column] = pd.to_numeric(df[column], errors='coerce')

            if "annual_fee" in df.columns:
                clean_currency_column(df, "annual_fee")

            required_columns = {"credit_card_name", "cardmaven_review", "cardexpert_review", "best_tags", "annual_fee"}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise DataLoadingError(f"Missing columns: {missing}")
            
            # Create combined column for RAG embedding
            df["combined_content"] = df.apply(
                lambda row: (
                    f"Card: {row['credit_card_name']}\n"
                    f"Reviews: {row['cardmaven_review'] or ''} {row['cardexpert_review'] or ''}\n"
                    f"Tags: {row['best_tags'] or ''}"
                ).strip(),
                axis=1
            )
            return df
        except Exception as e:
            raise DataLoadingError(f"Failed to load data: {str(e)}")

    def _setup_vectorstore(self) -> FAISS:
        """Set up FAISS vector store with embeddings of combined content.

        Returns:
            FAISS vector store instance.

        Raises:
            VectorStoreError: If vector store setup fails.
        """
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            if not self.faiss_index_dir.exists():
                # Load combined content with metadata
                loader = DataFrameLoader(
                    self.df,
                    page_content_column="combined_content"
                )
                docs = loader.load()
                if not docs:
                    raise VectorStoreError("No documents loaded from DataFrame.")
                
                # Add metadata for filtering
                for doc, row in zip(docs, self.df.itertuples()):
                    doc.metadata = {
                        "credit_card_name": row.credit_card_name,
                        "annual_fee": float(row.annual_fee) if pd.notna(row.annual_fee) else 99999
                    }
                
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = splitter.split_documents(docs)
                vectorstore = FAISS.from_documents(split_docs, embeddings)
                vectorstore.save_local(str(self.faiss_index_dir))
            else:
                vectorstore = FAISS.load_local(
                    str(self.faiss_index_dir), embeddings, allow_dangerous_deserialization=True
                )
            return vectorstore
        except Exception as e:
            raise VectorStoreError(f"Failed to set up vector store: {str(e)}")

    def _setup_llm(self) -> ChatOpenAI:
        """Initialize the language model.

        Returns:
            Configured ChatOpenAI instance.
        """
        return ChatOpenAI(
            openai_api_key=self.api_key,
            openai_api_base=OPENROUTER_API_BASE,
            model=LLM_MODEL,
        )

    def _setup_agent(self):
        """Initialize the LangChain agent with tools.

        Returns:
            Initialized agent.

        Raises:
            CreditCardAssistantError: If agent setup fails.
        """
        try:
            tools = [
                Tool(
                    name="get_card_details_by_name",
                    func=self.get_card_details_by_name,
                    description="Retrieve details for a specific credit card by name."
                ),
                Tool(
                    name="compare_cards_tool",
                    func=self.compare_cards,
                    description="Compare two credit cards. Format: 'Card1 and Card2' or 'Card1 vs Card2'."
                ),
                Tool(
                    name="get_best_cards_by_tag",
                    func=lambda tag: str(self.get_best_cards_by_tag(tag)),
                    description="Fetch raw data of top cards by category (e.g., 'best_cashback')."
                ),
                Tool(
                    name="rag_lookup",
                    func=self.rag_lookup,
                    description=(
                        "Answer queries about card features, benefits, or vague questions "
                        "(e.g., 'cards with lounge access', 'best cards for dining'). "
                        "Use this for semantic searches across reviews and tags."
                    )
                ),
                Tool(
                    name="summarize_best_cards",
                    func=self.summarize_best_cards,
                    description="Summarize top cards with rankings and recommendations."
                ),
            ]
            return initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
            )
        except Exception as e:
            raise CreditCardAssistantError(f"Failed to initialize agent: {str(e)}")

    def get_card_details_by_name(self, card_name: str) -> Union[Dict, str]:
        """Retrieve details for a specific credit card.

        Args:
            card_name: Name of the credit card.

        Returns:
            Dictionary of card details or error message.
        """
        try:
            card_name = card_name.strip().lower()
            matches = self.df[self.df["credit_card_name"].str.lower().str.strip() == card_name]
            if not matches.empty:
                return matches.iloc[0].to_dict()

            partial_matches = self.df[
                self.df["credit_card_name"].str.lower().str.contains(card_name, na=False)
            ]
            if not partial_matches.empty:
                best_match = partial_matches.iloc[
                    (partial_matches["credit_card_name"].str.len() - len(card_name)).abs().argmin()
                ]
                logger.info(f"Partial match found: {best_match['credit_card_name']}")
                return best_match.to_dict()

            words = card_name.split()
            token_matches = self.df[
                self.df["credit_card_name"].str.lower().apply(
                    lambda x: all(word in x for word in words)
                )
            ]
            if not token_matches.empty:
                best_match = token_matches.iloc[
                    (token_matches["credit_card_name"].str.len() - len(card_name)).abs().argmin()
                ]
                logger.info(f"Token match found: {best_match['credit_card_name']}")
                return best_match.to_dict()

            names = self.df["credit_card_name"].str.lower().tolist()
            close = difflib.get_close_matches(card_name, names, n=1, cutoff=0.4)
            if close:
                match_row = self.df[self.df["credit_card_name"].str.lower() == close[0]]
                logger.info(f"Fuzzy match found: {close[0]}")
                return match_row.iloc[0].to_dict()

            logger.warning(f"No card found for '{card_name}'.")
            return f"No card found for '{card_name}'."
        except Exception as e:
            logger.error(f"Error retrieving card details for '{card_name}': {str(e)}")
            return f"Error retrieving card details: {str(e)}"

    def compare_cards(self, query: str) -> Dict[str, Union[Dict, str]]:
        """Compare two credit cards.

        Args:
            query: Comparison query (e.g., 'Card1 vs Card2').

        Returns:
            Dictionary with details of both cards or error message.
        """
        try:
            pattern = re.compile(r"\s*(?:and|vs|versus)\s*", re.IGNORECASE)
            cards = pattern.split(query)
            if len(cards) != 2:
                return "Invalid format. Use 'Card1 and Card2' or 'Card1 vs Card2'."
            card1, card2 = [card.strip() for card in cards]
            return {
                "card_1": self.get_card_details_by_name(card1),
                "card_2": self.get_card_details_by_name(card2),
            }
        except Exception as e:
            logger.error(f"Error comparing cards: {str(e)}")
            return f"Error comparing cards: {str(e)}"

    def get_best_cards_by_tag(self, tag: str, max_fee: Optional[int] = None, top_n: int = 5) -> List[Dict]:
        """Retrieve top cards for a given tag.

        Args:
            tag: Category tag (e.g., 'best_travel').
            max_fee: Maximum annual fee filter (optional).
            top_n: Number of cards to return (default: 5).

        Returns:
            List of card dictionaries or error message.
        """
        try:
            filtered = self.df[self.df["best_tags"].str.contains(tag, case=False, na=False)]
            if not filtered.empty:
                if max_fee is not None:
                    filtered = filtered[filtered["annual_fee"].fillna(99999) <= max_fee]
                return filtered.head(top_n).to_dict(orient="records")

            all_tags = {
                tag.strip().lower()
                for tags in self.df["best_tags"].dropna()
                for tag in str(tags).split(",")
            }
            tag_words = set(tag.replace("best_", "").split("_"))
            best_match, best_score = None, 0
            for known_tag in all_tags:
                known_words = set(known_tag.replace("best_", "").split("_"))
                overlap = len(tag_words & known_words)
                if overlap > best_score or (overlap == best_score and tag in known_tag):
                    best_score = overlap
                    best_match = known_tag

            if best_match:
                filtered = self.df[
                    self.df["best_tags"].str.contains(best_match, case=False, na=False)
                ]
                if not filtered.empty:
                    if max_fee is not None:
                        filtered = filtered[filtered["annual_fee"].fillna(99999) <= max_fee]
                    logger.info(f"Using best-overlap tag: {best_match}")
                    return filtered.head(top_n).to_dict(orient="records")

            close = difflib.get_close_matches(tag, list(all_tags), n=1, cutoff=0.5)
            if close:
                filtered = self.df[
                    self.df["best_tags"].str.contains(close[0], case=False, na=False)
                ]
                if not filtered.empty:
                    if max_fee is not None:
                        filtered = filtered[filtered["annual_fee"].fillna(99999) <= max_fee]
                    logger.info(f"Using fuzzy-matched tag: {close[0]}")
                    return filtered.head(top_n).to_dict(orient="records")

            return [{"error": f"No cards found for tag '{tag}'."}]
        except Exception as e:
            logger.error(f"Error retrieving cards for tag '{tag}': {str(e)}")
            return [{"error": f"Error retrieving cards: {str(e)}"}]

    def rag_lookup(self, query: str) -> str:
        """Answer queries about card features or benefits using RAG.

        Args:
            query: User query (e.g., 'cards with lounge access').

        Returns:
            Answer or error message.
        """
        try:
            logger.info(f"Executing RAG query: {query}")
            response = self.qa_chain.invoke({"query": query})["result"]
            logger.info(f"RAG response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error processing RAG query '{query}': {str(e)}")
            return f"Error processing query: {str(e)}"

    def summarize_best_cards(self, input_data: Union[str, Dict, List]) -> str:
        """Summarize top cards with rankings and recommendations.

        Args:
            input_data: Card data (stringified list/dict or direct list/dict).

        Returns:
            Formatted summary string.
        """
        try:
            import ast
            tag = "unknown"
            cards = []

            if isinstance(input_data, str):
                try:
                    data = ast.literal_eval(input_data)
                except Exception as e:
                    return f"Error parsing input data: {str(e)}"
            else:
                data = input_data

            if isinstance(data, dict):
                tag = data.get("tag", "unknown")
                cards = data.get("cards", [])
            elif isinstance(data, list):
                cards = data
            else:
                return "Error: Invalid input format."

            if not cards or (isinstance(cards, dict) and "error" in cards):
                return f"No cards found for tag '{tag}'."

            def clean_card(card: Dict) -> Dict:
                return {k: v for k, v in card.items() if k in ALLOWED_CARD_FIELDS}

            def parse_fee(fee: Union[str, float, int]) -> float:
                if isinstance(fee, str):
                    fee = re.sub(r"[₹, +GST]", "", fee).strip()
                    try:
                        return float(fee)
                    except ValueError:
                        return 99999
                return float(fee) if isinstance(fee, (int, float)) else 99999

            def parse_reward_rate(review: str) -> float:
                if isinstance(review, str):
                    match = re.search(r"reward rate:?\s*(\d+\.?\d*%\s*-\s*\d+\.?\d*%)", review, re.IGNORECASE)
                    if match:
                        low, high = [float(x.strip("%")) for x in match.group(1).split("-")]
                        return (low + high) / 2
                return 1.0

            def parse_lounge_access(review: str) -> float:
                if isinstance(review, str):
                    if "unlimited" in review.lower() and "lounge" in review.lower():
                        return 3
                    if "lounge access" in review.lower():
                        return 2
                return 1

            def parse_luxury_perks(review: str) -> float:
                if not isinstance(review, str):
                    return 0
                perks = sum(
                    1 for perk in ["concierge", "golf", "accor", "marriott", "taj epicure"]
                    if perk in review.lower()
                )
                return perks

            def parse_welcome_bonus(review: str) -> float:
                if isinstance(review, str):
                    match = re.search(r"welcome benefit:?\s*(\d{1,3}(?:,\d{3})*)\s*(?:points|inr)", review, re.IGNORECASE)
                    if match:
                        return float(match.group(1).replace(",", ""))
                return 0

            scored_cards = []
            for card in [clean_card(c) for c in cards]:
                fee = parse_fee(card.get("annual_fee", "99999"))
                review = card.get("cardmaven_review", "") or card.get("cardexpert_review", "")
                reward_rate = parse_reward_rate(review)
                lounge_score = parse_lounge_access(review)
                perks_score = parse_luxury_perks(review)
                bonus_score = parse_welcome_bonus(review)

                fee_score = max(0, 1 - (fee / 50000))
                reward_score = min(reward_rate / 20, 1)
                lounge_score = lounge_score / 3
                perks_score = perks_score / 3
                bonus_score = min(bonus_score / 20000, 1)

                total_score = sum(
                    SCORING_WEIGHTS[key] * score
                    for key, score in [
                        ("annual_fee", fee_score),
                        ("reward_rate", reward_score),
                        ("lounge_access", lounge_score),
                        ("luxury_perks", perks_score),
                        ("welcome_bonus", bonus_score),
                    ]
                )
                scored_cards.append({
                    "name": card.get("credit_card_name", "Unknown"),
                    "score": total_score,
                    "annual_fee": fee,
                    "reward_rate": reward_rate,
                    "lounge_access": lounge_score * 3,
                    "luxury_perks": perks_score * 3,
                    "welcome_bonus": bonus_score * 20000,
                })

            scored_cards.sort(key=lambda x: x["score"], reverse=True)
            summary = f"Top {len(scored_cards)} Premium Credit Cards for '{tag}':\n\n"
            for i, card in enumerate(scored_cards, 1):
                summary += (
                    f"{i}. {card['name']}\n"
                    f"   - Annual Fee: ₹{card['annual_fee']:,.0f}\n"
                    f"   - Reward Rate: ~{card['reward_rate']:.1f}%\n"
                    f"   - Lounge Access: {'Unlimited' if card['lounge_access'] >= 3 else 'Limited' if card['lounge_access'] >= 2 else 'None'}\n"
                    f"   - Luxury Perks: {int(card['luxury_perks'])}/3\n"
                    f"   - Welcome Bonus: ~₹{card['welcome_bonus']:,.0f}\n"
                    f"   - Score: {card['score']:.2f}/1.00\n\n"
                )

            top_card = scored_cards[0]
            summary += (
                f"Recommendation:\n"
                f"The {top_card['name']} (score: {top_card['score']:.2f}) is the best choice, "
                f"offering a {top_card['reward_rate']:.1f}% reward rate, "
                f"{'unlimited' if top_card['lounge_access'] >= 3 else 'limited'} lounge access, "
                f"and {int(top_card['luxury_perks'])} luxury perks. "
                f"Its ₹{top_card['annual_fee']:,.0f} fee suits high spenders."
            )
            if len(scored_cards) > 1 and scored_cards[1]["score"] > 0.8 * top_card["score"]:
                summary += (
                    f" The {scored_cards[1]['name']} (score: {scored_cards[1]['score']:.2f}) "
                    f"is a strong alternative for lower fees or specific perks."
                )
            return summary
        except Exception as e:
            logger.error(f"Error summarizing cards: {str(e)}")
            return f"Error summarizing cards: {str(e)}"

    def run(self, query: str) -> str:
        """Process a user query.

        Args:
            query: User input query.

        Returns:
            Response from the agent.
        """
        try:
            return self.agent.invoke({"input": query})["output"]
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            return f"Error processing query: {str(e)}"

def main():
    """Run the CreditCardAssistant in interactive mode."""
    try:
        assistant = CreditCardAssistant(DATA_FILE, FAISS_INDEX_DIR)
        logger.info("Credit Card Assistant is ready. Type 'exit' or 'quit' to stop.")
        while True:
            query = input("Ask about credit cards: ")
            if query.lower() in ["exit", "quit"]:
                break
            response = assistant.run(query)
            print(f"\nResponse:\n{response}")
    except CreditCardAssistantError as e:
        logger.error(f"Initialization failed: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()