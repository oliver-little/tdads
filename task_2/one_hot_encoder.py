#one hot encoder

class Encoder:

    def __init__(self, file):
        self.max_tweet_length = 0
        self.dict_key = 0
        self.word_dictionary = {}
        self.tweets = file

    def find_max_tweet(self):
        current_max = 0
        for row in self.tweets.text:
            words = row.split(' ')

            if len(words) > current_max:
                current_max = len(words)

        self.max_tweet_length = current_max

    def dict_contains(self, word):

        for i in self.word_dictionary:
            if self.word_dictionary[i] == word:
                return False
        return True

    def create_word_dictionary(self):
        #key should be the number assigned to it
        for row in self.tweets.text:
            words = row.split(' ')
            for word in words:
                not_in = self.dict_contains(word)
                if not_in:
                    self.word_dictionary[self.dict_key] = word
                    self.dict_key += 1

    def find_key(self, word):

        for i in self.word_dictionary:
            if self.word_dictionary[i] == word:
                return i
        return -1;

    def convert_tweet(self, tweet):

        tweet_vector = []
        index = 0
        for x in range(self.max_tweet_length):
            tweet_vector.append(0)

        words = tweet.split(' ')
        for word in words:
            tweet_vector[index] = self.find_key(word)
            index += 1

        return tweet_vector
