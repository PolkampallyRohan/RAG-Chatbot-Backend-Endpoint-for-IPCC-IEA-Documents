import requests

API_URL = "http://127.0.0.1:8000/chat/rag"


def test_query(query: str):
    response = requests.post(API_URL, json={"query": query})
    response.raise_for_status()

    data = response.json()

    print("\nQuery:")
    print(data["query"])

    print("\nAnswer:")
    print(data["answer"])

    print("\nRetrieved chunks:")
    for chunk in data["retrieved_chunks"]:
        print(f"- {chunk['source']}: {chunk['content']}")


if __name__ == "__main__":
    test_query("How can renewable energy help reduce emissions?")
    test_query("What does climate adaptation involve?")