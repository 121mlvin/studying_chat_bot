import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

greetings = ['Hi!', 'Hello!', 'What\'s up', 'Howdy!']
goodbyes = ['Bye!', 'See ya!', 'Goodbye!']

data = {
    'music': 'Music is so relaxing!',
    'pet': 'Dogs are man\'s best friends',
    'book': 'I know about a lot of books',
    'game': 'I like to play video games'
}

keywords = list(data.keys())
responses = list(data.values())


def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


preprocessed_keywords = [preprocess(keyword) for keyword in keywords]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_keywords)
y = keywords

clf = MultinomialNB()
clf.fit(X, y)

print(random.choice(greetings))

while True:
    user = input('Say something! (or type bye to quit): ')
    if user.lower() == 'bye':
        print(random.choice(goodbyes))
        break

    user_processed = preprocess(user)
    user_vector = vectorizer.transform([user_processed])

    predicted_keyword = clf.predict(user_vector)[0]
    response_index = keywords.index(predicted_keyword)
    print('Bot: ' + responses[response_index])

    if predicted_keyword not in user_processed:
        new_keyword = input('I\'m not sure what to respond. What keyword should I use? ')
        keywords.append(new_keyword)
        new_response = input('How should I respond to "' + new_keyword + '"? ')
        responses.append(new_response)

        new_data = preprocess(new_keyword)
        X = vectorizer.fit_transform(keywords)
        y = keywords
        clf.fit(X, y)
