from collections import Counter, defaultdict

SENTENCES = [
    'nvidia corporation is a market leader among technology companies specializing in graphics processing units gpus and artificial intelligence ai', 
    'the company is renowned for its innovative approaches to both hardware and software solutions', 
    'it reported strong earnings in the most recent quarter', 
    'it showcased robust revenue growth driven primarily by increasing demand for its gpus across various sectors including gaming data centers and professional visualization', 
    'additionally nvidia ai focused initiatives continued to gain traction with expanding applications in sectors such as autonomous vehicles healthcare and cloud computing', 
    'despite facing supply chain challenges and global semiconductor shortages nvidia managed to exceed market expectations demonstrating resilience and adaptability in navigating the volatile economic landscape', 
    'as a result investors responded positively to the earnings report indicating confidence in nvidia longterm growth prospects'
]

TAG_SENTENCES = [
    ["NNP","NN","VBZ","DT","NN","NN","IN","NN","NNS","VBG","IN","NNS","NN","NNS","NNS","CC","JJ","NN","NN"],
    ["DT","NN","VBZ","JJ","IN","PRP$","JJ","NNS","IN","DT","NN","CC","NN","NNS"],
    ["PRP","VBD","JJ","NNS","IN","DT","RBS","JJ","NN"],
    ["PRP","VBD","JJ","NN","NN","VBN","RB","IN","VBG","NN","IN","PRP$","NNS","IN","JJ","NNS","VBG","NN","NN","NNS","CC","JJ","NN"],
    ["RB","NNP","NN","JJ","NNS","VBD","TO","VB","NN","IN","VBG","NNS","IN","NNS","JJ","IN","JJ","NNS","NN","CC","NN","NN"],
    ["IN","VBG","NN","NN","NNS","CC","JJ","NN","NNS","NNP","VBD","TO","VB","NN","NNS","VBG","NN","CC","NN","IN","VBG","DT","JJ","JJ","NN"],
    ["IN","DT","NN","NNS","VBD","RB","IN","DT","NNS","NN","VBG","NN","IN","NNP","JJ","NN","NNS"]
]

def get_start_probs(tag_sentences) -> dict[str, float]:
    # Count appearances of each tag at the start of sentences
    start_counts = Counter(sentence[0] for sentence in tag_sentences)
    total = sum(start_counts.values())
    return {tag: count / total for tag, count in start_counts.items()}

def get_transition_probs(tag_sentences) -> dict[str, dict[str, float]]:
    # Count the number of each pair of tags
    transition_counts = defaultdict(Counter)
    for sentence in tag_sentences:
        for prev_tag, curr_tag in zip(sentence[:-1], sentence[1:]):
            transition_counts[prev_tag][curr_tag] += 1

    # Get the probability of each transition for each tag
    transition_probs = {}
    for prev_tag, counter in transition_counts.items():
        total = sum(counter.values())
        transition_probs[prev_tag] = {
            curr_tag: count / total for curr_tag, count in counter.items()
        }
    return transition_probs

def get_emission_probs(token_sentences, tag_sentences) -> dict[str, dict[str, float]]:
    # Count each emission
    emission_counts = defaultdict(Counter)
    for words, tags in zip(token_sentences, tag_sentences):
        for word, tag in zip(words, tags):
            emission_counts[tag][word] += 1

    # Get the probability of each emission for each tag
    emission_probs = {}
    for tag, counter in emission_counts.items():
        total = sum(counter.values())
        emission_probs[tag] = {
            word: count / total for word, count in counter.items()
        }
    return emission_probs

def viterbi(words, pos_tags, start_prob, transition_prob, emission_prob):
    """
    words: list of words from the paragraph
    pos_tags: list of possible POS tags
    start_prob: Start probs (from above functions)
    transition_prob: Transition probs (from above functions)
    emission_prob: Emission probs (from above functions)
    """
    # Dynamic programming table for tracking the best probabilities
    prob_table = [{}]
    
    # Keeps track of the actual path to get the best probability (from the table)
    path = {}
    
    # Initialization (for the first word)
    for tag in pos_tags:
        prob_table[0][tag] = start_prob.get(tag, 0) * emission_prob[tag].get(words[0], 0)
        path[tag] = [tag]

    # Recursion step
    for t in range(1, len(words)):
        prob_table.append({})
        new_path = {}

        for curr_tag in pos_tags:
            # Get the max probability (I added a small epsilon here to prevent sparse data from causing issues)
            prob, prev_tag = max(
                (prob_table[t-1].get(prev_tag, 0) * transition_prob.get(prev_tag, {}).get(curr_tag, 0) * (emission_prob[curr_tag].get(words[t], 0) + 1e-6), prev_tag) 
                for prev_tag in pos_tags
            )
            # Write it to the table and keep track of the path
            prob_table[t][curr_tag] = prob
            new_path[curr_tag] = path[prev_tag] + [curr_tag]

        path = new_path

    # Get the best probability/path from the table
    (best_prob, best_tag) = max([(prob_table[-1][tag], tag) for tag in pos_tags])
    return (best_prob, path[best_tag])


def main():
    # Get the word tokens from the paragraph
    tokens = [[word for word in sentence.split()] for sentence in SENTENCES]
    
    # All used POS tags
    pos_tags = ("NNP", "NN", "NNS", "VB", "VBD", "VBZ", "VBG", "DT", "JJ", "RB", "IN", "CC", "PRP", "PRP$")

    # Calculate probability tables for the paragraph words/tokens
    start_probs = get_start_probs(TAG_SENTENCES)
    transition_probs = get_transition_probs(TAG_SENTENCES)
    emission_probs = get_emission_probs(tokens, TAG_SENTENCES)

    print(emission_probs)

    # This is the new sentence
    new_sentence = "nvidia is a leader among technology companies"

    # Run the Viterbi algorithm
    print(viterbi(new_sentence.split(), pos_tags, start_probs, transition_probs, emission_probs))

if __name__ == "__main__":
    main()