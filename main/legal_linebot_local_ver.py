import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from lib.Utils import GetRoot
from lib.Utils import GetHfToken
from lib.Utils import GetLineAccess
from lib.Utils import GetLineSecret
from lib.RAG import ResponseWithJudgement


# Load environment variables
GetRoot()

# Add HuggingFace token
token_info = GetHfToken()
print(token_info) if token_info is not None else print(">>> HuggingFace token NOT found.")

# Obtain the embedding files
fil_embeddings = os.path.join(
    os.environ.get("PROJECT_ROOT") + os.getenv("DIR_DATA"),
    "embeddings",
    "laws_embedding_4chunk_2overlap_Llama.json"
)

# Line bot deployment
LINE_CHANNEL_ACCESS_TOKEN = GetLineAccess()
LINE_CHANNEL_SECRET = GetLineSecret()
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Flask application
app = Flask(__name__)

# Line webhook
# 定義 Flask 的 HTTP 路由，將來自 Line 的 POST 請求映射到 callback 函數
@app.route("/", methods=["POST"])
# Line Webhook 路由，接收来自 Line 平台的 Webhook 請求
def callback():
    # 驗證Line簽名，確定請求是由 Line 伺服器發出
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        # 將請求內容傳送給 handler 處理
        handler.handle(body, signature)
    except InvalidSignatureError:
        # 驗證失敗回傳 HTTP 400 error
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
# 事件處理
def handle_message(event):
    try:
        # 接收用戶訊息
        user_message = event.message.text
        print(">>> 成功接收用戶訊息")

        # Llama產生回答
        response = ResponseWithJudgement(query=user_message)

        # 返回回答給 Line 用戶
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response)
        )
    except Exception as e:
        print(f">>> 處理流程發生錯誤: {str(e)}")


if __name__ == "__main__":
    app.run(port=5000)
