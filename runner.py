from pprint import pprint
from GraphTopic import GraphTopic
from data.prepare import get_sample_data


def main():
    docs = get_sample_data()

    model = GraphTopic(docs, TopM=200, TopN=10, overlapping_threshold=0.5)
    pprint(model.get_topics())


if __name__ == "__main__":
    main()
