
'''
File: sentiment_analysis.py
Author: Kragen Wild
Date: April 15, 2024

This provides sentiment analysis functions for processing tweets in particular,
but relies on tweet_processing to handle the cleanup of the tweets. Analysis is
done using Naive Bayes.

'''
import random
import tweet_processor as tp
import math
from openai import OpenAI

import numpy as np

def partition_training_and_test_sets(pos_tweets : list[str],
                                     neg_tweets : list[str], 
                                     split : float = .8) -> tuple[list[str], 
                                                                  np.ndarray[float], 
                                                                  list[str], 
                                                                  np.ndarray[float], 
                                                                  int, int, int, int]:
    '''
    Partition our sets of tweets into positive and negative tweets based
    on a split factor. 

    Parameters: 
        pos_tweets -- list of strings that are positive tweets
        neg_tweets -- list of strings that are negative tweets
        split -- factor to split the training and partition sets into.
            Defaults to .8, or 80% training, 20% testing.

    Returns:
        A list of training tweets
        A list of the same size of training labels, which will be 1 or 0 for positive or negative tweets
        A list of testing tweets
        A list of testing labels, which 
    '''
    if split < 0 or split > 1:
        raise Exception('split must be between 0 and 1')
    
    # multiply the length of the list of tweets by our split factor and convert to an int
    pos_train_size = int(split * len(pos_tweets))
    neg_train_size = int(split * len(neg_tweets))
    
    # split our sets
    pos_x = pos_tweets[:pos_train_size]
    neg_x = neg_tweets[:neg_train_size]
    
    # test sets
    test_pos = pos_tweets[pos_train_size:]
    test_neg = neg_tweets[neg_train_size:]

    # combine the sets for training and testing
    train_x = pos_x + neg_x
    test_x = test_pos + test_neg

    # our labels are 1 for positive, 0 for negative, so we'll create
    # arrays of 1s and 0s for the training and test sets
    train_y = np.append(np.ones(len(pos_x)), np.zeros(len(neg_x)))
    test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

    pos_test_size = len(pos_tweets) - pos_train_size
    neg_test_size = len(neg_tweets) - neg_train_size
    return (train_x, train_y, test_x, test_y, pos_train_size, 
            neg_train_size, pos_test_size, neg_test_size)


# takes a list of tweets        
def build_word_freq_dict(tweets : list[list[str]], labels : np.ndarray[int]) -> dict[(str, int),  int]:
    '''
    Creates a frequency dictionary based on the tweets. The frequency dictionary
    has keys which are (word, label) pairs, for example, ('happi', 1), while the
    value associated with it is the number of times that word was seen in a given
    class. For example, if 'happi' is seen 10 times in positive tweets, then we'd 
    see freqs[('happi', 1)] = 10. If it were seen 3 times in negative tweets, we'd
    see freqs[('happi', 0)] = 3.

    Parameters: 
    tweets -- A list of strings, each a tweet
    labels -- A list of integers either 0 or 1 for negative or positive classes

    Note that the number of tweets and labels must match. 

    Return: 
    A dictionary containing (word, class) keys mapping to the number of 
    times that word in that class appears in the data set
    '''
    dict = {}
    vocab = set()
    
    # create the dictionary and vocabulary here
    # for every tweet...
    for i in range(len(tweets)):
        # for every word...
        for j in range(len(tweets[i])):
            # word is added to the vocab set
            vocab.add(tweets[i][j])

            # if the word and label pair is not yet in the dictionary, its value is set to 1
            if (tweets[i][j], labels[i]) not in dict.keys():
                dict[(tweets[i][j], int(labels[i]))] = 1
            # if the word and label pair is in the dictionary, its value is incimented by 1
            else:
                dict[(tweets[i][j], int(labels[i]))] += 1




    # return the frequency dictionary
    # print(f'vocab: {vocab}')
    # print(f'dict: {dict}')
    return dict, vocab

def test_word_freq_dict():
    '''
    Simple function that tests some tweets and if your build_word_freq_dict is built correctly
    '''
    tweets = [['i', 'am', 'happi'], ['i', 'am', 'trick'], ['i', 'am', 'sad'], 
              ['i', 'am', 'tire'], ['i', 'am', 'tire']]
    labels = [1, 0, 0, 0, 0]
    print("testing build_word_freq_dict, should get {('i', 1): 1, ('am', 1): 1, ('happi', 1): 1, ('i', 0): 4, ('am', 0): 4, ('trick', 0): 1, ('sad', 0): 1, ('tire', 0): 2}")
    print(f'test of word frequency: {build_word_freq_dict(tweets, labels)}')


def count_pos_neg(freqs : dict[(str, int),  int]) -> tuple[int, int]:
    '''
    Count the number of positive and negative words in the
    frequency dictionary.

    Parameters:
    freqs -- a dictionary of ((str, int), int) pairs, where the key is a
    word and label of 0 or 1 for negative or positive sentiment, and the value
    associated with the key is the number of times it was seen.

    Returns:
    Returns two values, the number of times any positive word was seen (i.e., the
    total number of positive events), and the number of times a negative word was
    seen. 
    '''
    num_pos = num_neg = 0
    for key in freqs.keys():
        if key[1] == 0:
            num_neg += 1
        elif key[1] == 1:
            num_pos += 1
        else:
            print("SOMETHING IS VERY VERY WRONG")

    return num_pos, num_neg



def build_loglikelihood_dict(freqs : dict[(str, int),  int], N_pos : int, N_neg : int, vocab : list[str]) -> dict[str, float]:
    '''
    Create a dictionary based on the frequency of each word in each class appearing
    of the probability of that word occuring, using Laplacian smoothing by adding
    1 to each occurrence and the size of the vocabulary. 

    Thus, we'd calculate (freq(w_i, class) + 1) / (N_class + V_size)

    Parameters:
        freqs -- dictionary from (word, class) to occurrence count mapping
        N_pos -- number of positive events for all words
        N_neg -- number of negative events for all words
        vocab -- list vocabulary of words

    Returns:
        A dictionary of words to the ratio of positive and negative usage of the word
    '''
    loglikelihood = {}
    vocab_size = len(vocab)

    # calculate the loglikelihood dictionary from the given parameters
    for word in vocab:
        if (word, 1) not in freqs.keys():
            pos_prob = 1/(N_pos + vocab_size)
        else:
            pos_prob = (freqs[(word, 1)] + 1)/(N_pos + vocab_size)
        if (word, 0) not in freqs.keys():
            neg_prob = 1/(N_neg + vocab_size)
        else:
            neg_prob = (freqs[(word, 0)] + 1)/(N_neg + vocab_size)
        # TODO: REVIEW THIS MATH
        loglikelihood[word] = math.log(pos_prob/neg_prob)
        

    return loglikelihood

def naive_bayes_predict(loglikelihood : dict[str, float], log_pos_neg_ratio : float, tweet : list[str]) -> float:
    '''
    Calculates the prediction based on our dictionary of log-likelihoods of each
    word in a tweet added to the log of the ratio of positive and negative tweets

    Parameters:
        loglikelihood -- a dictionary of words to the ratio of postive/negative probabilities of the words
        log_pos_neg_ratio -- the log of the ratio of total positive to total negative events
        tweet -- a list of tokens (likely from process_tweet)
    '''
    # Return the prediction of a given tweet using the dictionary and ratio
    # TODO: The build loglikelihood dictionary function might not be the one to do the logarithms counulations, this one might be
    returnable = 0
    for word in tweet:
        if word in loglikelihood.keys():
            returnable += (loglikelihood[word])
        else:
            returnable += 0
    returnable += log_pos_neg_ratio
    return returnable

# def collect_messages(client, chatbox_context) -> str:
#     prompt = input("User> ")
#     chatbox_context.append({"role": "user", "content": prompt})
#     response = get_msg_completion(client, chatbox_context).choices[0].message.content
#     print(f"Assistant> {response}")
#     chatbox_context.append({"role": "assistant", "content": response})
#     return prompt

# def get_msg_completion(client: OpenAI, messages, temperature: float = 0,
#                        model : str = 'gpt-3.5-turbo') -> str:
    
#     '''
#     Parameters:
#     client -- an instance of the OpenAI class
#     messages -- a list of strings, each a string is a message
#     temperature -- a float representing the randomness of the completion
#     model -- a string representing the model to use for the completion

#     Returns:
#     A string representing the completion of the messages
#     '''
#     response = client.chat.completions.create(
#         model = model,
#         messages = messages,
#         temperature = temperature
#     )
#     return response




def test_word_freq_dict(tweets, labels):
    '''
    Simple function that tests some tweets and if your build_word_freq_dict is built correctly
    '''
    print(f'test of word frequency: {build_word_freq_dict(tweets, labels)}')

def test_loglikelihood_dict(tweets, labels):
    '''
    Simple function that tests some tweets and there result of build_loglikelihood_dict
    
    '''
    word_freqs, vocab = build_word_freq_dict(tweets, labels)
    print(f'word_freqs: {word_freqs}')
    print(f'vocab: {vocab}')
    num_pos, num_neg = count_pos_neg(word_freqs)
    print(f'pos: {num_pos}, neg: {num_neg}')
    loglikelihood_dict = build_loglikelihood_dict(word_freqs, num_pos, num_neg, vocab)
    print(f'testing build_loglikelihood_dict, should get {loglikelihood_dict}')
    
def debug_sentiment_analysis(tweets : list[str], stopwords : list[str], labels : np.array) -> None:
    """
    Testing of some sentiment analysis functions
    """
    # create a list of processed tweets
    processed_tweets = [tp.process_tweet(tweet, stopwords) for tweet in tweets]

    test_word_freq_dict(processed_tweets, labels)

    test_loglikelihood_dict(processed_tweets, labels)

def main():
    # first, set up our samples
    pos_tweets, neg_tweets, stopwords, full_pos_tweets, full_neg_tweets = tp.process_tweets('SentimentAnalysis/positive_tweets.json', 'SentimentAnalysis/negative_tweets.json', 'SentimentAnalysis/english_stopwords.txt')
    
    # you can uncomment the next two lines once your tweet processing is working
    print(f'random positive: {pos_tweets[random.randint(0, len(pos_tweets) - 1)]}')
    print(f'random negative: {neg_tweets[random.randint(0, len(neg_tweets) - 1)]}')

    # defines the partition between training and test sets
    SPLIT = .8
    
    # next, partition the sets into training sets, test sets, and labels
    train_x, train_y, test_x, test_y, N_train_pos, N_train_neg, N_test_pos, N_test_neg = partition_training_and_test_sets(pos_tweets, neg_tweets, SPLIT)
    print(f'N_train_pos = {N_train_pos}, N_train_neg = {N_train_neg}')
    print(f'N_test_pos = {N_test_pos}, N_test_neg = {N_test_neg}')

    # create a frequency dictionary
    freq_train, vocab = build_word_freq_dict(train_x, train_y)
    # print(f'Freqs: {freq_train}')
    # print(f'Vocab {(vocab)}')
    print(f'freq dictionary size: {len(freq_train)}, vocab size: {len(vocab)}')
    
    # count the number of positive and negative words
    num_pos, num_neg = count_pos_neg(freq_train)
    print(f'Number of positive events: {num_pos}, Number of negative events: {num_neg}')

    # log of the ratio of the total positive and total negative tweets from the training set
    log_pos_neg_ratio = math.log(num_pos/num_neg)
    print(f'log_pos_neg_ratio of the training set = {log_pos_neg_ratio}')

    # now calculate the log likelihood dictionary
    log_likelihood = build_loglikelihood_dict(freq_train, num_pos, num_neg, vocab)
    

    # process the tweets
    # debugging
    # debug_train_tweets = full_pos_tweets[:10] + full_neg_tweets[:10]
    # debug_labels = np.append(np.ones(10), (np.zeros(10)))
    # # now debug them to see the output
    # debug_sentiment_analysis(debug_train_tweets, stopwords, debug_labels)

    # uncomment the code below once you have everything above working
    # now let's test some predictions
    for i in range(10):
        idx = random.randint(0, N_test_pos + N_test_neg - 1)
        print(f'Tweet: {test_x[idx]}')
        print(f'Label: {test_y[idx]}')
        print(f'Prediction: {naive_bayes_predict(log_likelihood, log_pos_neg_ratio, test_x[idx])}')
        print()

    # now let's see what our error rate is
    # Calculate the error rate and print it out
    # also print out the mislabeled tweets
    
    print('\nCalculating error rate...\n')
    mislabeledTweets = {}
    # format: {tweet number: {tweet: tweet, label: label, bayes prediction: bayes prediction}}
    for i in range(len(test_x)):
        if int(test_y[i]) == 0 and naive_bayes_predict(log_likelihood, log_pos_neg_ratio, test_x[i]) > 0:
            tweet_number = len(mislabeledTweets) + 1
            mislabeledTweets[f"Tweet {tweet_number}"] = {}
            mislabeledTweets[f"Tweet {tweet_number}"]["tweet"] = test_x[i]
            mislabeledTweets[f"Tweet {tweet_number}"]["label"] = (test_y[i])
            mislabeledTweets[f"Tweet {tweet_number}"]["bayes prediction"] = naive_bayes_predict(log_likelihood, log_pos_neg_ratio, test_x[i])
        elif int(test_y[i]) == 1 and naive_bayes_predict(log_likelihood, log_pos_neg_ratio, test_x[i]) < 0:
            tweet_number = len(mislabeledTweets) + 1
            mislabeledTweets[f"Tweet {tweet_number}"] = {}
            mislabeledTweets[f"Tweet {tweet_number}"]["tweet"] = test_x[i]
            mislabeledTweets[f"Tweet {tweet_number}"]["label"] = test_y[i]
            mislabeledTweets[f"Tweet {tweet_number}"]["bayes prediction"] = naive_bayes_predict(log_likelihood, log_pos_neg_ratio, test_x[i])

    # prints out the mislabeled tweets
    for key, value in mislabeledTweets.items():
            if int(value["label"]) == 1:
                print(f"Human thought {key} was positive while bayes thought it was negative")
            elif int(value["label"]) == 0:
                print(f"Human thought {key} was negative while bayes thought it was positive")
            for key2, value2 in value.items():

                print(f'{key2}: {value2}')
            print("\n")


    error_rate = len(mislabeledTweets)/len(test_x)
    print(f'Error rate: {error_rate}\n')


    # creates a list of just the mislabeled tweets
    just_tweets = [' '.join(value["tweet"]) for value in mislabeledTweets.values()]
    
    #  creates a list of the human responses
    human_responses = []
    for value in mislabeledTweets.values():
        if int(value["label"]) == 1:
            human_responses.append("positive sentiment")
        elif int(value["label"]) == 0:
            human_responses.append("negative sentiment")

    
    

    # asks the AI to interpret the sentiment of the mislabeled tweets, storing its response in a list
    client = OpenAI()
    
    completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                messages=[{"role": "system", "content": """You are a linguistic expert, able to decern tweets' sentiment. You are also an expert at navigating python lists.
                                                           You are goint to be given a python list of tweets. Your job is to interpret the sentiment of the tweet and decide whether it's postitve or negative."""
                                                           },
                                                          {"role": "user", "content": f"""For each tweet in the list, delimited by triple backtick, perform the following actions:                                                          
                                                           Step 1 - Interpret the tweet and decide whether it is postitve or negative by responding with either "postitve sentiment" or "negative sentiment" in all lowercase. "neutral sentiment" is not a valid response.
                                                           Step 2 - Repeat step 1 for the next tweet in the list, but respond on a new line
                                                           '''{just_tweets}'''"""},])
    print("\nLLM's opinion on the mislabeled tweets:\n")
    AI_responses = completion.choices[0].message.content.split("\n")
    for i in range(len(AI_responses)):
        print(f"Tweet: {just_tweets[i]}")
        print(f"Sentiment: {AI_responses[i]}\n")


    # creates a dictionary of the disagreements between the AI and the human
    # format: {tweet: AI response}
    disagreement = {}
    for i in range(len(AI_responses)):
        if human_responses[i] != AI_responses[i]:
            disagreement[just_tweets[i]] = AI_responses[i]

    # asks the AI to explain why it thinks the way it does about the mislabeled tweets
    completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                messages=[{"role": "system", "content": """You are a linguistic expert, able to decern tweets' sentiment. You are also an expert at navigating python disctionaries.
                                                           You are goint to be given a dictionary, with each key consisting of a tweet, and each value consisting of that tweet's sentiment. Your job is to expalain why each tweet is catogorized with the seniment it's given."""
                                                           },
                                                          {"role": "user", "content": f"""For each tweet in the nested dictionary, delimited by triple backtick, perform the following actions:
                                                           Step 1 - Explain why the tweet is catogorized with the sentiment it's given. Respond in a conversational manner.
                                                           '''{disagreement}'''"""},])
    print("\nLLM's explanation of tweets it disagreed with the human label on:\n")
    print(completion.choices[0].message.content)
    print("\n")

        

    # writing my own tweet to fool the LLM


    my_tweet = "People who are so happy they are always smiling and talking about how fun and beautiful their lives are are the worst"

    print(f"\nMy attempt to fool the LLM: {my_tweet}\n")
    print(f"Bayes prediction: {naive_bayes_predict(log_likelihood, log_pos_neg_ratio, tp.process_tweet(my_tweet, stopwords))}\n")


    completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                messages=[{"role": "system", "content": """You are a linguistic expert, able to decern tweets' sentiment.
                                                           You are goint to be given a tweet, and your job is to explain it's sentiment as either positive or negative."""
                                                           },
                                                          {"role": "user", "content": f"""
                                                           Step 1 - Determine if the tweet "{my_tweet}" is positive or negative, and respond with either "positive sentiment" or "negative sentiment." 
                                                           Step 2 - Explain your reasoning."""},])
    print(f"LLM response:\n{completion.choices[0].message.content}")












# run the main function if this is where our program was executed from
if __name__ == '__main__':
    main()