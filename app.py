from flask import Flask, render_template, request
from cbot import TranslatorBot

app = Flask(__name__)
bot = TranslatorBot()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def translate():
    print("Translate function called!")
    text = request.form['text']
    source_lang = request.form['source_lang']
    target_lang = request.form['target_lang']
    translated_text = bot.translate(text, source_lang, target_lang)
    return render_template('index.html', translated_text=translated_text)


if __name__ == '__main__':
    app.run(port=4000, debug=True)
