from flask import Flask, request, jsonify
import random
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

corpus = """
Cricket is a bat-and-ball game that is played between two teams of eleven players on a field at the centre of which is a 22-yard (20-metre) pitch with a wicket at each end, each comprising two bails balanced on three stumps. Two players from the batting team (the striker and nonstriker) stand in front of either wicket, with one player from the fielding team (the bowler) bowling the ball towards the striker's wicket from the opposite end of the pitch. The striker's goal is to hit the bowled ball and then switch places with the nonstriker, with the batting team scoring one run for each exchange. Runs are also scored when the ball reaches or crosses the boundary of the field or when the ball is bowled illegally.

The fielding team tries to prevent runs from being scored by dismissing batters (so they are "out"). Means of dismissal include being bowled, when the ball hits the striker's wicket and dislodges the bails, and by the fielding side either catching the ball after it is hit by the bat, but before it hits the ground, or hitting a wicket with the ball before a batter can cross the crease in front of the wicket. When ten batters have been dismissed, the innings ends and the teams swap roles. Forms of cricket range from Twenty20 (also known as T20), with each team batting for a single innings of 20 overs (each "over" being a set of 6 fair opportunities for the batting team to score) and the game generally lasting three to four hours, to Test matches played over five days.

Traditionally cricketers play in all-white kit, but in limited overs cricket they wear club or team colours. In addition to the basic kit, some players wear protective gear to prevent injury caused by the ball, which is a hard, solid spheroid made of compressed leather with a slightly raised sewn seam enclosing a cork core layered with tightly wound string.

The earliest known definite reference to cricket is to it being played in South East England in the mid-16th century. It spread globally with the expansion of the British Empire, with the first international matches in the second half of the 19th century. The game's governing body is the International Cricket Council (ICC), which has over 100 members, twelve of which are full members who play Test matches. The game's rules, the Laws of Cricket, are maintained by Marylebone Cricket Club (MCC) in London. The sport is followed primarily in South Asia, Australia, New Zealand, the United Kingdom, Southern Africa and the West Indies.[1]
"""

def tokenize(text):
    return nltk.sent_tokenize(text), nltk.word_tokenize(text)

def lemmatize(tokens):
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def get_response(user_query, corpus):
    bot_response = ''
    sent_tokens, word_tokens = tokenize(corpus)
    sent_tokens.append(user_query)
    tfidf_obj = TfidfVectorizer(tokenizer=lemmatize, stop_words='english')
    tfidf = tfidf_obj.fit_transform(sent_tokens)
    sim_values = cosine_similarity(tfidf[-1], tfidf)
    index = sim_values.argsort()[0][-2]
    flattened_sim = sim_values.flatten()
    flattened_sim.sort()
    required_tfidf = flattened_sim[-2]
    if required_tfidf == 0:
        bot_response += 'I cannot understand'
        return bot_response
    else:
        bot_response += sent_tokens[index]
        return bot_response

@app.route('/api/chatbot', methods=['GET'])
def chatbot():
    user_query = request.args.get('query')
    if not user_query:
        return jsonify({'error': 'Query parameter is required'}), 400
    response = get_response(preprocess(user_query), corpus)
    return jsonify({'response': response}), 200

if __name__ == '__main__':
    app.run(debug=True)
