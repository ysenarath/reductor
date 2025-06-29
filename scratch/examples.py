from __future__ import annotations

from reductor.agents.classifier import ClassifierAgent

if __name__ == "__main__":
    clf = ClassifierAgent(
        target_type="multiclass_classifier",
        classes=[
            "sports",
            "politics",
            "entertainment",
            "technology",
            "health",
        ],
    )
    docs = [
        "The team won the championship in a thrilling match.",
        # "The government passed a new law to improve healthcare.",
        # "A new movie is breaking box office records.",
        # "The latest smartphone features cutting-edge technology.",
        # "A new study reveals the benefits of a balanced diet.",
        # "The athlete broke the world record in the 100m sprint.",
        # "The president addressed the nation on economic reforms.",
        # "A popular TV show is returning for a new season.",
    ]
    output = clf.predict_proba(docs)
    print(output)
