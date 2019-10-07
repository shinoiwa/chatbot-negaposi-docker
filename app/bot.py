from flask import Flask, request, abort

import os
import numpy as np
import pickle

from janome.tokenizer import Tokenizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

app = Flask(__name__)

channel_secret = os.environ['LINE_CHANNEL_SECRET']
channel_access_token = os.environ['LINE_CHANNEL_ACCESS_TOKEN']

# LineBotApiとWebhookへの接続
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

def wakati(text):
    tokenizer = Tokenizer()
    word_list = []

    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(",")[0]
        if pos in ["名詞", "動詞", "形容詞"]:
            lemma = token.base_form
            word_list.append(lemma)
    textrep = u" ".join(word_list)
    return textrep

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):

    wakatis = []
    for line in open('wordList2.txt','r',encoding="utf_8"):
        wakatis.append(line)

    # 書き込みを単語ベクトルに変換
    count_vectorizer = CountVectorizer()
    feature_vectors = count_vectorizer.fit_transform(wakatis)  # csr_matrix(疎行列)が返る

    # SVMモデルのロード
    classifier = svm.SVC(C = 10, gamma = 0.1, kernel = 'rbf')
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # 新しいデータに対する分類
    new_text = wakati(event.message.text)
    new_post_vec = count_vectorizer.transform([new_text])
    predict_label_new = loaded_model.predict(new_post_vec)

    if predict_label_new[0]=='1':
        textrep = "よかったね"
    else:
        textrep = "ざんねん"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(extrep))

if __name__ == "__main__":
    #port = int(os.getenv("PORT"))
    #app.run(host="0.0.0.0", port=port)
    app.run()
