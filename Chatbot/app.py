from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load your CSV data
data = pd.read_csv('BCG-TES.csv')

# Define chatbot response logic
def chatbot_response(user_input):
    if user_input.lower() == "stock price":
        return "The stock price is 20000."
    elif user_input.lower() == "market news":
        return "Today, the market is steady with mixed trends in the tech sector."
    elif user_input.lower() == "show data":
        return data.head().to_html(index=False)
    elif user_input.lower() == "goodbye":
        return "Goodbye! Have a great day!"
    else:
        return "Sorry, I didn't understand that. Try: 'stock price', 'market news', 'show data', or 'goodbye'."

# Web routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response():
    user_input = request.args.get('msg')
    response = chatbot_response(user_input)
    return str(response)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
