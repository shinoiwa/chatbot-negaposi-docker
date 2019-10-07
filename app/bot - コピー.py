from flask import Flask, request, abort

import os

from janome.tokenizer import Tokenizer

from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import pickle

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

    # svmのテスト

    # アヤメデータセットを用いる
    iris = datasets.load_iris()

    # 例として、3,4番目の特徴量の2次元データで使用
    X = iris.data[:, [2,3]]
    #クラスラベルを取得
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None )

    # データの標準化処理
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # 線形SVMのインスタンスを生成
    model = SVC(kernel='linear', random_state=None)

    # モデルの学習。fit関数で行う。
    model.fit(X_train_std, y_train)

    # トレーニングデータに対する精度
    #loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    #pred_train = loaded_model.predict(X_train_std)
    pred_train = model.predict(X_train_std)
    accuracy_train = accuracy_score(y_train, pred_train)

    t = Tokenizer()

    textrep = ''
    for token in t.tokenize(event.message.text):
        textrep += token.base_form + "\t" + token.part_of_speech + "\n"

    textrep += str(accuracy_train)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=textrep))

if __name__ == "__main__":
    #port = int(os.getenv("PORT"))
    #app.run(host="0.0.0.0", port=port)
    app.run()
