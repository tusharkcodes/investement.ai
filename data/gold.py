from langchain_community.document_loaders import WebBaseLoader
from textsplitter.split_document import split_document_chunks


loader = WebBaseLoader([
    "https://groww.in/p/gold-investment",
    "https://groww.in/p/sgb-vs-mutual-funds",
    "https://groww.in/p/sovereign-gold-bonds-vs-physical-gold",
    "https://groww.in/p/sovereign-gold-bond-vs-gold-etf",
    "https://groww.in/p/savings-schemes/gold-savings-scheme",
    "https://www.investopedia.com/articles/basics/08/gold-strategies.asp",
    "https://www.jpmorgan.com/insights/global-research/commodities/gold-prices"
])

docs = loader.load()

chunks = split_document_chunks(
    documents=docs,
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Number of chunks created: {len(chunks)}")

