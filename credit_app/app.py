from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def credit_risk_assessment(income, debt, credit_score, loan_amount, employment_years):
    """Simple heuristic credit risk assessment model (Score: 0-100)."""
    debt_to_income_ratio = debt / income if income > 0 else 1
    credit_score_factor = (700 - credit_score) / 7  # Normalized inverse factor
    loan_to_income_ratio = loan_amount / income if income > 0 else 1
    employment_factor = max(0, 5 - employment_years) * 5

    risk_score = (
        debt_to_income_ratio * 40 +
        credit_score_factor * 30 +
        loan_to_income_ratio * 20 +
        employment_factor * 10
    )

    return min(max(risk_score, 0), 100)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/assess', methods=['POST'])
def assess():
    data = request.form
    income = float(data.get('income', 0))
    debt = float(data.get('debt', 0))
    credit_score = float(data.get('credit_score', 650))
    loan_amount = float(data.get('loan_amount', 0))
    employment_years = float(data.get('employment_years', 0))


    risk_score = credit_risk_assessment(income, debt, credit_score, loan_amount, employment_years)
    
    return jsonify({'risk_score': round(risk_score, 2)})

if __name__ == '__main__':
    app.run(debug=True)
