import collections
import numpy as np
from math import log

EPSILON = 1e-5
START_TAG = "START"
END_TAG = "END"


def smoothed_prob(arr, alpha=1):
    #checking if its not numpy array , converting it to one 
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    #calculating the sum of the array
    _sum = arr.sum()
    #checking if the sum is non zero 
    if _sum:
        #applying the laplace (add-one) smoothing  and convert back to list
        return ((arr + alpha) / (_sum + arr.size * alpha)).tolist()
    else:
        #handling the case  when the sum is zero to avoaid divison by zero
        return ((arr + 1) / arr.size).tolist()


def baseline(train, test):
    #counter for tag occurences
    tag_counts = collections.Counter()
    #defaultdict for word  tag occuerences
    word_tag_counts = collections.defaultdict(collections.Counter)

    # Count occurrences of tags and tag-word pairs in the training data
    for sentence in train:
        for word, tag in sentence:
            tag_counts[tag] += 1
            word_tag_counts[word][tag] += 1

    #finding the most common tag in the training data
    most_common_tag = tag_counts.most_common(1)[0][0]
   #list to  store predicted_sentencss
    predicted_sentences = []

    #predicting  tags for test data
    for sentence in test:
        #again list to storre  predicted sentences 
        predicted_tags = []
        for word in sentence:
            #checking if the word was seen in training data
            if word in word_tag_counts:
                #finding the most common tag for the word
                predicted_tag = max(word_tag_counts[word], key=word_tag_counts[word].get)
            else:
                #if the word is unseen, use the most common tag in training data
                predicted_tag = most_common_tag
            predicted_tags.append(predicted_tag)
        #combinig words  and predicted tags into tuples n adding to the list
        predicted_sentence = list(zip(sentence, predicted_tags))
        predicted_sentences.append(predicted_sentence)
    #returning the  list  of predicted sentences 
    return predicted_sentences

def viterbi(train, test):
    #defaultdics for transition  and emission counts
    tag_transition_counts = collections.defaultdict(lambda: collections.Counter())
    tag_emission_counts = collections.defaultdict(lambda: collections.Counter())
    #counter for tag occurences
    tag_counts = collections.Counter()

    #counting transitions and emissions in the training data
    for sentence in train:
        prev_tag = START_TAG
        for word, tag in sentence:
            tag_counts[tag] += 1
            tag_emission_counts[tag][word] += 1
            tag_transition_counts[prev_tag][tag] += 1
            prev_tag = tag
        tag_transition_counts[prev_tag][END_TAG] += 1

    #calculating initial probabilities
    initial_probs = {tag: (tag_transition_counts[START_TAG][tag] + EPSILON) / 
                     (sum(tag_transition_counts[START_TAG].values()) + EPSILON * len(tag_counts))
                     for tag in tag_counts}

    #calculating transition probabilities
    transition_probs = {
        prev_tag: {tag: (tag_transition_counts[prev_tag][tag] + EPSILON) / 
                   (sum(tag_transition_counts[prev_tag].values()) + EPSILON * len(tag_counts))
                   for tag in tag_counts}
        for prev_tag in tag_counts
    }

    #calculating emission probabilities
    emission_probs = {
        tag: {word: (tag_emission_counts[tag][word] + EPSILON) / 
              (tag_counts[tag] + EPSILON * len(tag_emission_counts[tag]))
              for word in tag_emission_counts[tag]}
        for tag in tag_counts
    }
    #list to store predictde sentencs
    predicted_sentences = []
    #iterating through the test sentences
    for sentence in test:
        #list to store to the viterbi a nd backpointer matrices
        viterbi_matrix = [{}]
        backpointer_matrix = [{}]

        #initializing step with safe log probability access
        for tag in tag_counts:
            #calculating the initial probability for the current tag (with Laplace smoothing)
            initial_prob = max(initial_probs.get(tag, EPSILON), EPSILON)  #ensure non-zero
            #calculating the emission probability for the first word in the sentence (with Laplace smoothing)
            emission_prob = max(emission_probs[tag].get(sentence[0], EPSILON), EPSILON)  # Ensure non-zero
            #calculating the log probability and store it in the Viterbi matrix
            viterbi_matrix[0][tag] = log(initial_prob) + log(emission_prob)
            #intializing the backpointer matrix for the fisrt  position with none 
            backpointer_matrix[0][tag] = None

        #recursion step
        for t in range(1, len(sentence)):
            viterbi_matrix.append({})
            backpointer_matrix.append({})

            for current_tag in tag_counts:
                #calculatin the maximum probabliity and corresponding  previous tag
                max_prob, max_tag = max(
                    #list comprehenmsion to calculate the score for each previous tag
                    [(viterbi_matrix[t-1][prev_tag] + log(transition_probs[prev_tag].get(current_tag, EPSILON)) +
                      log(emission_probs[current_tag].get(sentence[t], EPSILON)), prev_tag)
                     for prev_tag in tag_counts],
                    key=lambda x: x[0]
                )
                #store the maximum probability and corresponding previous tag in the viteerbi matrix 
                viterbi_matrix[t][current_tag] = max_prob
                backpointer_matrix[t][current_tag] = max_tag

        #termination step
        last_probs = [(viterbi_matrix[-1][tag] + log(transition_probs[tag].get(END_TAG, EPSILON)), tag)
                      for tag in tag_counts]
        max_prob_final, final_tag = max(last_probs, key=lambda x: x[0])

        #backtracking to fingd the most likely sequence of tags 
        predicted_tags = [final_tag]
        for t in range(len(sentence) - 1, 0, -1):
            predicted_tags.insert(0, backpointer_matrix[t][predicted_tags[0]])
        #combimig the words and predicted atgg into tuples and adiding to the list  
        predicted_sentence = list(zip(sentence, predicted_tags))
        predicted_sentences.append(predicted_sentence)
    #returning  to the list of predicted sentences 
    return predicted_sentences
