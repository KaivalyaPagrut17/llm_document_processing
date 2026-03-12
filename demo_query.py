from sentence_transformers import SentenceTransformer
import chromadb

# simple demo script that returns the top 3 clauses (dummy output)
# replace with actual search logic if desired later

query = input("Enter your query: ")

print("\nTop relevant clauses:\n")

# dummy results to make the demo smooth
print("1. Knee surgery requires a 24 month waiting period.")
print("2. Orthopedic procedures coverage conditions.")
print("3. Policy waiting period rules.")
