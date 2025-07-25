from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from .forms import RegisterForm
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import cv2
from django.conf import settings
import os
from django.contrib.auth.decorators import login_required,user_passes_test
from django.core.files.storage import FileSystemStorage
import numpy as np
from PIL import Image

def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')

def query(request):
    return render(request,'appointment.html')

def sessions(request):
    return render(request,'classes.html')

def contact(request):
    return render(request,'contact.html')

def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('login_view')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('detect')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout1(request):
    logout(request)
    messages.success(request, "Logged out successfully")
    return redirect('/')


@login_required(login_url='/login_view/')  # Redirects to the login page if not logged in
def detect(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_path)

        # Load and preprocess image
        img = Image.open(fs.path(file_path))
        img = img.resize((64, 64))
        img_array = np.array(img)

        # Convert to Grayscale
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray_path = os.path.join(fs.location, "gray_" + uploaded_file.name)
        cv2.imwrite(gray_path, gray_img)

        # Convert to Binary (Thresholding)
        _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_path = os.path.join(fs.location, "binary_" + uploaded_file.name)
        cv2.imwrite(binary_path, binary_img)
        
        from keras.models import load_model
        # Load the trained CNN model
        model = load_model(r'Diy_model.h5',compile=True)

        # Prepare for model prediction
        img_array = img_array.reshape(1, 64, 64, 3).astype('float32') / 255.0
        prediction = model.predict(img_array)
        disease_class = np.argmax(prediction)
        print("disease_class: ",disease_class)
        # Assign disease label
        if disease_class == 0:
            result = "Normal Person"
        elif disease_class == 1:
            result = "Dyslexic Disease Person"
        else:
            result = "Unknown Condition"

        return render(request, 'detect.html', {
            'file_url': file_url,
            'gray_url': fs.url("gray_" + uploaded_file.name),
            'binary_url': fs.url("binary_" + uploaded_file.name),
            'result': result
        })

    return render(request, 'detect.html')


import numpy as np
from joblib import dump, load
from django.shortcuts import render
from joblib import load
from django.http import FileResponse
from reportlab.pdfgen import canvas
from datetime import datetime
import os
from django.conf import settings
from sklearn.svm import SVC


@login_required(login_url='/login_view/')  # Redirects to the login page if not logged in
def predict_dyslexia(request):
    result = None
    entered_data = {}
    pdf_url = None
    recommendations = []

    if request.method == "POST":
        try:
            # Get user details
            username = request.user.username if request.user.is_authenticated else "Guest"
            email = request.user.email if request.user.is_authenticated else "N/A"

            # Get form data
            entered_data = {
                # "Attendance": int(request.POST.get("Attendance")),
                "Confidence": int(request.POST.get("Confidence")),
                "Participation": int(request.POST.get("Participation")),
                "Health Issues": "No" if request.POST.get("Health_Issues") == "No" else "Yes",
                "Distraction": request.POST.get("Distraction"),
                "Pronunciation Issues": request.POST.get("Pronunciation_Issues"),
                "Reading Fluency": int(request.POST.get("Reading_Fluency")),
                "Writing Legibility": int(request.POST.get("Writing_Legibility")),
                "Math Struggles": "No" if request.POST.get("Math_Struggles") == "No" else "Yes",
                "Memory Issues": request.POST.get("Memory_Issues"),
            }

            # Convert categorical values to numeric
            e2, e3 = entered_data["Confidence"], entered_data["Participation"]
            e4 = 0 if entered_data["Health Issues"] == "No" else 1
            e5 = 0 if entered_data["Distraction"] == "Always" else 2
            e6 = 0 if entered_data["Pronunciation Issues"] == "Always" else 2
            e7, e8 = entered_data["Reading Fluency"], entered_data["Writing Legibility"]
            e9 = 0 if entered_data["Math Struggles"] == "No" else 1
            e10 = {"Always": 0, "Sometimes": 1, "Never": 2}[entered_data["Memory Issues"]]

            # Load trained model
            model = load(r"model.joblib")

            # Predict Dyslexia
            prediction = model.predict([[e2, e3, e4, e5, e6, e7, e8, e9, e10]])
            result = "Dyslexia Predicted" if prediction[0] == 1 else "Normal Person"
            

            # Generate recommendations based on the prediction
            if prediction[0] == 1:
                recommendations = [
                    "Encourage phonics-based reading programs.",
                    "Provide additional time for reading and writing tasks.",
                    "Use multi-sensory teaching techniques (visual, auditory, kinesthetic).",
                    "Provide structured routines and a distraction-free environment.",
                    "Use assistive technologies like text-to-speech software.",
                    "Encourage confidence-building activities and positive reinforcement."
                ]
            else:
                recommendations = [
                    "Continue regular learning habits and exercises.",
                    "Encourage participation in interactive learning activities.",
                    "Maintain confidence through self-paced learning.",
                    "Use structured study techniques for better retention.",
                    "Encourage curiosity and creativity in reading and writing."
                ]

            # Generate PDF Report
            pdf_filename = f"dyslexia_report_{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            pdf_filepath = os.path.join(settings.MEDIA_ROOT, pdf_filename)

            p = canvas.Canvas(pdf_filepath)
            p.setFont("Helvetica", 12)

            # Title
            p.drawString(200, 800, "Dyslexia Prediction Report")

            # User Info
            p.drawString(50, 770, f"Username: {username}")
            p.drawString(50, 750, f"Email: {email}")
            p.drawString(50, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Prediction Result
            p.drawString(50, 700, f"Prediction: {result}")

            # Entered Data
            y_position = 670
            for key, value in entered_data.items():
                p.drawString(50, y_position, f"{key}: {value}")
                y_position -= 20

            # Recommendations
            y_position -= 30
            p.drawString(50, y_position, "Recommended Solutions:")
            y_position -= 20
            for rec in recommendations:
                p.drawString(70, y_position, f"- {rec}")
                y_position -= 20

            # Save PDF
            p.showPage()
            p.save()

            # Provide a downloadable link
            pdf_url = f"/media/{pdf_filename}"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render(request, "predict_text.html", {"result": result, "pdf_url": pdf_url, "recommendations": recommendations})

# PDF Download View (Not needed anymore because files are stored in MEDIA)

# PDF Download View
# def download_report(request):
    # buffer = BytesIO(request.session.get("pdf_buffer", b""))
    # return FileResponse(buffer, as_attachment=True, filename="dyslexia_report.pdf")