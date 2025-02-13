from flask import Flask,render_template

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/user')
def user():
    return render_template('user.html')

if __name__=='__main__':
    app.run(debug=True,port=5002)
