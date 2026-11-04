from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


def evaluate_rag(query, answer, contexts):
    """
    Evaluate RAG output using RAGAS metrics
    """

    # Ensure contexts are strings
    contexts = [str(c) for c in contexts]

    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts]
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision
        ]
    )

    return result