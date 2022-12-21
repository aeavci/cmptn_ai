from flask import Flask, render_template, request, redirect, session
import pickle
from model import myFunc


app = Flask(__name__)
app.secret_key = 'BAD_SECRET_KEY'
model = pickle.load(open("comp.pkl", "rb"))


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/feature")
def feature():
    return render_template("feature.html")


@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if request.method == 'POST':
        trc = 0
        source = request.form["Source"]
        if(source == 'Diamond Dataset'):
            session['my_var'] = 'dd'
            trc = 1
        elif(source == 'Car Price Prediction Dataset'):
            session['my_var'] = 'cp'
            trc = 2
        else:
            session['my_var'] = 'ab'
            trc = 3

        return render_template('visualize.html', tracker=trc)
    elif request.method == 'GET':
        return redirect('/')
    else:
        return 'Not a valid request method for this route'


@app.route('/scores', methods=['GET', 'POST'])
def scores():
    if request.method == 'POST':
        y = 0
        if('dd' == session['my_var']):
            y = 1
        elif('cp' == session['my_var']):
            y = 2
        else:
            y = 3
        mla = request.form["mla"]
        if(mla == 'Linear Regression'):
            x = myFunc(1, y)
        elif(mla == "Ridge Regression"):
            x = myFunc(7, y)
        elif(mla == "Elastic Net Regression"):
            x = myFunc(8, y)
        elif(mla == 'Decision Tree Regression'):
            x = myFunc(2, y)
        elif(mla == "KNN"):
            x = myFunc(3, y)
        elif(mla == "Random Forest Regression"):
            x = myFunc(4, y)
        elif(mla == "Gradient Boosting Regression"):
            x = myFunc(5, y)
        elif(mla == "Support Vector Regression"):
            x = myFunc(6, y)
        else:
            x = myFunc(9, y)
        # make an object named users consisting of 5 random users and their random scores
        users = [{'name': 'John', 'score': 97.15}, {'name': 'Mary', 'score': 88.71}, {'name': 'Bob', 'score': 73.22},
                 {'name': 'Jane', 'score': 66.15}, {'name': 'Joe', 'score': 55.71}]
    



        


        
      
        return render_template('scores.html', prediction_text="Ihr Genauigkeitswert ist: {}".format(x), users=users)
    elif request.method == 'GET':
        return redirect('/')
    else:
        return 'Not a valid request method for this route'


if __name__ == '__main__':
    app.run(debug=True)
