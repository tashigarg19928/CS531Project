import logging
import os
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Form, UploadFile, File
from fastapi import Request
from passlib.context import CryptContext
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.staticfiles import StaticFiles
from werkzeug.utils import secure_filename
from fastapi.templating import Jinja2Templates

from ml_model import train_lstm_model, \
    predict_next_month_lstm, detect_anomalies, cluster_expenses, recommend_savings_plan, fetch_expense_data, \
    detect_anomalies_autoencoder
from models import get_user_by_username, get_user_by_id, create_user, create_expense, get_goals_by_user_id, create_goal, get_user_by_id, \
    update_user_profile, create_income, get_income_by_user_id, get_expenses_fortbl_by_user_id, init_db

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.secret_key = 'your_secret_key'
# Serve static files from the "static" folder
app.mount("/static", StaticFiles(directory="static"), name="static")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
templates = Jinja2Templates(directory="templates")
# Configure upload folder
UPLOAD_FOLDER = 'static/profile_images'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit


# Initialize MongoDB collections at startup
@app.on_event("startup")
async def startup_db():
    init_db()  # Ensure collections are created on app startup


# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

logging.basicConfig(level=logging.INFO)


# Function to hash a password
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# Function to verify a password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user_id = request.cookies.get('user_id')
    if user_id:
        user = get_user_by_id(user_id)
        model, scaler = train_lstm_model(user_id)
        next_month_prediction = predict_next_month_lstm(user_id, model, scaler)
        return templates.TemplateResponse("home.html", {"request": request, "prediction": next_month_prediction, "user": user})
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def get_login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def get_register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/login")
async def login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...)
):
    user = get_user_by_username(username)
    if user and verify_password(password, user["password"]):
        response = templates.TemplateResponse("home.html", {"request": request, "user": user})
        response.set_cookie(key="user_id", value=str(user["_id"]))
        return response
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


@app.post("/register")
async def register(
        request: Request,
        username: str = Form(...),
        password: str = Form(...)
):
    if get_user_by_username(username):
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already taken"})

    hashed_password = hash_password(password)
    is_user_created = create_user(username, hashed_password)

    if is_user_created:
        return templates.TemplateResponse("login.html", {"request": request, "message": "Registration successful!"})
    else:
        return templates.TemplateResponse("register.html", {"request": request, "message": "Error creating user"})


@app.get("/logout")
async def logout(request: Request):
    response = templates.TemplateResponse("login.html", {"request": request})
    response.delete_cookie("user_id")
    return response

@app.get("/add_expense", response_class=HTMLResponse)
async def get_add_expense_page(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Please login first"})
    user = get_user_by_id(user_id)
    return templates.TemplateResponse("add_expense.html", {"request": request, "user": user})

@app.post("/add_expense")
async def add_expense(
        request: Request,
        category: str = Form(...),
        amount: float = Form(...),
        date: str = Form(...),
        description: str = Form(None),
):
    user_id = request.cookies.get("user_id")
    if not user_id:
        response = RedirectResponse(url="/login", status_code=303)
        response.set_cookie(key="message", value="Please login first!", max_age=5)
        return response

    # Call create_expense to insert the data into MongoDB
    create_expense(user_id, category, amount, date, description)

    # Redirect to the view_expenses page with a success message
    response = RedirectResponse(url="/view_expenses", status_code=303)
    response.set_cookie(key="message", value="Expense added!", max_age=5)
    return response


@app.get("/view_expenses", response_class=HTMLResponse)
async def view_expenses(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Please login first"})

    expenses = get_expenses_fortbl_by_user_id(user_id)
    user = get_user_by_id(user_id)
    if not expenses:
        raise HTTPException(status_code=404, detail="No expenses found for this user.")
    return templates.TemplateResponse("view_expenses.html", {"request": request, "expenses": expenses, "user": user})

@app.get("/add_goal", response_class=HTMLResponse)
async def get_add_goal_page(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url=request.url_for("login"), status_code=303)

    return templates.TemplateResponse("add_goal.html", {"request": request})

@app.post("/add_goal", response_class=RedirectResponse)
async def add_goal(
        request: Request,
        goal: str = Form(...),
        target_amount: float = Form(...),
        current_amount: float = Form(...)
):
    user_id = request.cookies.get("user_id")  # Get the user_id from session
    if not user_id:
        return RedirectResponse(url="/login")

    create_goal(user_id, goal, target_amount, current_amount)
    #Any flash message????

    # Redirect the user to view goals after adding the goal
    return RedirectResponse(url="/view_goals", status_code=303)


@app.get("/view_goals", response_class=HTMLResponse)
async def view_goals(request: Request):
    user_id = request.cookies.get("user_id")  # Get the user_id from session
    if not user_id:
        return RedirectResponse(url="/login")

    # Retrieve the goals for the current user
    goals = get_goals_by_user_id(user_id)
    return templates.TemplateResponse("view_goals.html", {"request": request, "goals": goals})


@app.get("/profile")
async def profile(request: Request):
    # Simulate fetching the user from cookies (authentication)
    user_id = request.cookies.get("user_id")
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return templates.TemplateResponse("profile.html", {"request": request, "user": user})


@app.post("/profile")
async def update_profile(request: Request, new_username: str = Form(...), new_password: Optional[str] = Form(None),
                         profile_image: UploadFile = File(None)):
    user_id = request.cookies.get("user_id")  # Get user from cookies

    # Fetch current user data
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Handle password update (if new password is provided)
    hashed_password = None
    if new_password:
        hashed_password = pwd_context.hash(new_password)

    # Handle file upload
    filename = None
    if profile_image:
        filename = secure_filename(profile_image.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await profile_image.read())

    # Update user profile
    update_user_profile(user_id, new_username, hashed_password, filename)

    return RedirectResponse(url="/profile", status_code=303)


# Endpoint to add income
@app.get("/add_income", response_class=HTMLResponse)
async def get_add_income_page(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url=request.url_for("login"), status_code=303)

    return templates.TemplateResponse("add_income.html", {"request": request})

@app.post("/add_income", response_class=RedirectResponse)
async def add_income(
        request: Request,
        source: str = Form(...),
        amount: float = Form(...),
        date: str = Form(...)
):
    user_id = request.cookies.get("user_id")  # Get the user ID from session (cookie or other means)

    if not user_id:
        return RedirectResponse(url="/login")

    create_income(user_id, source, amount, date)
    #Any flash message???????

    # Redirect to the view income page after adding income
    return RedirectResponse(url="/view_income", status_code=303)


# Endpoint to view income
@app.get("/view_income")
async def view_income(request: Request):
    user_id = request.cookies.get("user_id")  # Get the user ID from session (cookie or other means)

    # Check if user_id is null or empty, and redirect to login page if not authenticated
    if not user_id:
        return RedirectResponse(url="/login")
    # Simulate retrieving income data from the database for a specific user
    income = get_income_by_user_id(user_id)

    # Render the view_income.html template with the incomes
    return templates.TemplateResponse("view_income.html", {"request": request, "income": income})


@app.get("/predict_expenses")
async def predict_expenses(request: Request):
    user_id = request.cookies.get("user_id")  # Authenticate user via cookies

    # Fetch user's expense data
    expenses = fetch_expense_data(user_id)
    if expenses.empty or expenses.shape[0] < 2:  # Check for sufficient historical data
        logging.warning(f"Not enough data available for user {user_id} to make a prediction.")
        return templates.TemplateResponse("predict_expenses.html", {"request": request,
                                                                    "error": "Not enough data available to make a prediction."})

    # Train the LSTM model
    model, scaler = train_lstm_model(user_id)
    if not model or not scaler:
        logging.error(f"Model training failed for user {user_id} due to insufficient data.")
        return templates.TemplateResponse("predict_expenses.html",
                                          {"request": request, "error": "Not enough data to train the model."})

    # Make the prediction
    next_month_prediction = predict_next_month_lstm(user_id, model, scaler)
    if next_month_prediction is None:
        logging.error(f"Prediction could not be made for user {user_id}")
        return templates.TemplateResponse("predict_expenses.html",
                                          {"request": request, "error": "Not enough data to make a prediction."})

    # Prepare data for rendering
    next_month_prediction = float(next_month_prediction)

    expenses['date'] = pd.to_datetime(expenses['date'])
    expenses.set_index('date', inplace=True)
    expenses = expenses.resample('ME').sum()  # Resample by month-end (ME)
    dates = expenses.index.strftime('%Y-%m').tolist()
    amounts = expenses['amount'].tolist()

    # Add the prediction to the data
    labels = dates + ['Next Month']
    actual_expenses = amounts + [None]
    predicted_expenses = [None] * len(amounts) + [next_month_prediction]

    # Return the template response with the data
    return templates.TemplateResponse("predict_expenses.html", {
        "request": request,
        "prediction": next_month_prediction,
        "labels": labels,
        "actual_expenses": actual_expenses,
        "predicted_expenses": predicted_expenses
    })


@app.get("/recommend_savings")
async def recommend_savings(request: Request):
    # Authenticate user based on cookies (or implement session checking here)
    user_id = request.cookies.get("user_id")  # Get the user ID from cookies/session

    # Get the recommended savings amount for the user
    recommended_amount = recommend_savings_plan(user_id)

    # Render the template with the recommended savings amount
    return templates.TemplateResponse("recommend_savings.html",
                                      {"request": request, "recommended_amount": recommended_amount})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
