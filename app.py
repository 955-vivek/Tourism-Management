from click import prompt
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
import os
import json
from transformers import AutoTokenizer
from huggingface_hub.inference._client import InferenceClient
from dotenv import load_dotenv
import requests
from urllib.parse import quote
from fake_useragent import UserAgent
import time
from tavily import TavilyClient

load_dotenv()

OpenTrip_API_KEY = os.getenv("Open_Trip_Map_API_KEY")
if not OpenTrip_API_KEY:
    raise ValueError("OpenTrip API key not found.")

HUGGINGFACE_API_KEY = os.getenv("huggingface_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise ValueError("Hugging Face API key not found.")

Tavily_API_KEY = os.getenv("Tavily_API_Key")
if not Tavily_API_KEY:
    raise ValueError("Tavily API key not found.")


# Hugging Face API configuration
HUGGINGFACE_API_TOKEN = HUGGINGFACE_API_KEY
HUGGINGFACE_MODEL = "facebook/bart-large-cnn"


app = Flask(__name__)

db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tourism.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SECRET_KEY'] = 'mysecret'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    reviews = db.relationship('Review', backref='author', lazy='dynamic')
    complaints = db.relationship('Complaint', backref='author', lazy='dynamic')

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    place_name = db.Column(db.String(100), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    review_text = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    complaint_text = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), default='Pending')
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Function to get dataset
def get_location_info_from_tavily(location_name, city_name=None, country_name=None):
    try:
        tavily_client = TavilyClient(api_key=Tavily_API_KEY)
        response = tavily_client.search(f"{location_name} as a tourist destination of city {city_name} in country {country_name}",)
        print(f"Response from Tavily: {response}")
        return response
    except Exception as e:
        return f"Error: {e}"

#get most popular tourist spots in a given destination using OpenTripMap API
def get_nearby_opentripmap(lat, lon):
    radius = 10000  # in meters
    limit = 10
    url = (
        f"https://api.opentripmap.com/0.1/en/places/radius?"
        f"radius={radius}&lon={lon}&lat={lat}&format=json&limit={limit}&apikey={OpenTrip_API_KEY}"
    )
    response = requests.get(url)
    places = response.json()
    return [(p["name"], p.get("kinds", ""), p["xid"]) for p in places if p["name"]]

# Summarize attractions using Hugging Face API
def summarize_attractions(destination, attractions):
    # Limit the number of attractions
    max_attractions = 10
    attractions = attractions[:max_attractions]

    summary_input = (
        f"List the top 10 attractions for someone visiting {destination} based on the list:\n\n"
        + "\n".join(attractions)
    )

    try:
        # Token-safe truncation using the same tokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        inputs = tokenizer(summary_input, return_tensors="pt", truncation=True, max_length=1024)
        input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        client = InferenceClient(
            provider="hf-inference",
            api_key=HUGGINGFACE_API_KEY,
        )

        result = client.summarization(
            input_text,
            model="facebook/bart-large-cnn",
        )
        print(f"Response from Hugging Face: {result}")

        if isinstance(result, dict) and 'summary_text' in result:
            return result['summary_text']
        elif isinstance(result, list) and 'summary_text' in result[0]:
            return result[0]['summary_text']

        return "Summary unavailable. Try again later."
    
    except Exception as e:
        return f"Error summarizing with Hugging Face: {str(e)}"


# Return latitude and longitude from a place name using Nominatim API
def get_lat_lon_from_place(place_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "TourismApp/1.0" 
    }
       
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    
    if data:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return lat, lon
    else:
        return None, None


# act as main function to summarize nearby places
def summarize_nearby_places(lat, lon, destination_name, city=None, country=None):
    locations = get_nearby_opentripmap(lat, lon)
    summarized_descriptions = []

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    client = InferenceClient(provider="hf-inference", api_key=HUGGINGFACE_API_KEY)


    for idx, (name, kinds, xid) in enumerate(locations, 1):
        print(f"\n🔎 {idx}. Getting details for: {name}")
        raw_desc = get_location_info_from_tavily(name, city, country)
        print(f"Raw description for {name}: {raw_desc}")

        if not raw_desc or "no description available" in str(raw_desc).lower():
            summarized = f"{name}: No description available to summarize."
        else:
            summary_input = f"Summarize this tourist attraction in 2-3 lines:\n\n{name}: {raw_desc}"
            try:
                inputs = tokenizer(summary_input, return_tensors="pt", truncation=True, max_length=1024)
                input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

                result = client.summarization(
                    text=input_text,
                    model="facebook/bart-large-cnn",
                )

                if isinstance(result, dict) and 'summary_text' in result:
                    summarized = result['summary_text']
                elif isinstance(result, list) and 'summary_text' in result[0]:
                    summarized = result[0]['summary_text']
                else:
                    summarized = raw_desc  # fallback

            except Exception as e:
                summarized = f"{name}: Error summarizing: {str(e)}"

        summarized_descriptions.append(f"{name}: {summarized}")
        #time.sleep(0.5)

    return summarized_descriptions


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect('/login')
        except IntegrityError:
            db.session.rollback()
            flash('Username or email already exists.', 'danger')
    return render_template('register.html')

# filepath: c:\Users\vivek\Downloads\Tourism-Updated-main\Tourism-Updated-main\app.py
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your username or password.', 'danger')
    return render_template('login.html')

@app.route('/review', methods=['GET', 'POST'])
@login_required
def review():
    if request.method == 'POST':
        place_name = request.form['place_name']
        rating = int(request.form['rating'])
        review_text = request.form['review_text']
        new_review = Review(place_name=place_name, rating=rating, review_text=review_text, user_id=current_user.id)
        db.session.add(new_review)
        try:
            db.session.commit()
            flash('Review submitted successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error submitting review: {str(e)}', 'danger')
        return redirect(url_for('index'))
    return render_template('review.html')

@app.route('/complaint', methods=['GET', 'POST'])
@login_required
def complaint():
    if request.method == 'POST':
        complaint_text = request.form['complaint_text']
        new_complaint = Complaint(complaint_text=complaint_text, user_id=current_user.id)
        db.session.add(new_complaint)
        try:
            db.session.commit()
            flash('Complaint submitted successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error submitting complaint: {str(e)}', 'danger')
        return redirect(url_for('index'))
    return render_template('complaint.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    recommendations = []
    if request.method == 'POST':
        destination = request.form['destination'].strip().title()
        city = request.form['city'].strip().title()
        country = request.form['country'].strip().title()
        if not destination or not city or not country:
            recommendations = ["Please provide a valid destination, city, and country."]
            return render_template('recommend.html', recommendations=recommendations)
        
        lat, lon = get_lat_lon_from_place(destination)
        if lat is None or lon is None:
            recommendations = [f"Error: Could not find coordinates for {destination}. Please check the spelling or try a different location."]
        else:
            summary = summarize_nearby_places(lat, lon, destination, city, country)
        recommendations.extend(summary)
    return render_template('recommend.html', recommendations=recommendations)

@app.route('/helplines')
def helplines():
    return render_template('helplines.html')

def init_db():
    with app.app_context():
        db.create_all()

init_db()

if __name__ == '__main__':
    app.run(debug=True)