
import psycopg2
from flask import Flask, render_template  
app = Flask(__name__)
@app.route('/')
def home():
  return render_template('homepage.html')
  
@app.route('/Our-services')
def services():
  return render_template('services.html')

@app.route('/Book-consultation')
def book():
  return render_template( 'index.html')

@app.route('/callender')
def calender():
  return render_template('calender.html')
  
if __name__=='__main__':
  app.run(host='0.0.0.0',debug=True)