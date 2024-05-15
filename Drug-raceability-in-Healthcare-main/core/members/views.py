from django.utils import timezone
import io
from reportlab.pdfgen.canvas import Canvas
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
import plotly.graph_objs as go
from django.db.models import Count
from django.db.utils import OperationalError
from django.db import connection
from .models import Member, Medicine_basic
from django.conf import settings
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import os
import pickle
import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Assuming the file is in the same directory
model_path = "C:/Users/arige/OneDrive/Desktop/major/core/static/paracetemol.pkl"

# Verify file existence (optional)
if not os.path.exists(model_path):
    print(f"Error: File '{model_path}' not found.")
    exit()  # Or handle the error differently

# Print the path for debugging
print(f"Loading model from: {model_path}")

try:
    with open(model_path, 'rb') as file:
        paracetemol_model = pickle.load(file)
except FileNotFoundError as e:
    print(f"Error: Failed to open file '{model_path}'. ({e})")
except pickle.UnpicklingError as e:
    print(f"Error: Error unpickling the model. ({e})")
else:
    # Model loaded successfully
    print("Model loaded successfully!")


# Assuming the file is in the same directory
aspirin_path = "C:/Users/arige/OneDrive/Desktop/major/core/static/Aspirin.pkl"

# Verify file existence (optional)
if not os.path.exists(aspirin_path):
    print(f"Error: File '{aspirin_path}' not found.")
    exit()  # Or handle the error differently

# Print the path for debugging
print(f"Loading model from: {aspirin_path}")

try:
    with open(aspirin_path, 'rb') as file:
        aspirin_model = pickle.load(file)
except FileNotFoundError as e:
    print(f"Error: Failed to open file '{aspirin_path}'. ({e})")
except pickle.UnpicklingError as e:
    print(f"Error: Error unpickling the model. ({e})")
else:
    # Model loaded successfully
    print("Aspirin Model loaded successfully!")

amlopidine_path = "C:/Users/arige/OneDrive/Desktop/major/core/static/Amlopidine.pkl"

# Verify file existence (optional)
if not os.path.exists(amlopidine_path):
    print(f"Error: File '{amlopidine_path}' not found.")
    exit()  # Or handle the error differently

# Print the path for debugging
print(f"Loading model from: {amlopidine_path}")

try:
    with open(amlopidine_path, 'rb') as file:
        amlopidine_model = pickle.load(file)
except FileNotFoundError as e:
    print(f"Error: Failed to open file '{amlopidine_path}'. ({e})")
except pickle.UnpicklingError as e:
    print(f"Error: Error unpickling the model. ({e})")
else:
    # Model loaded successfully
    print("Amlopidine Model loaded successfully!")


lisinopril_path = "C:/Users/arige/OneDrive/Desktop/major/core/static/Lisinopril.pkl"

# Verify file existence (optional)
if not os.path.exists(lisinopril_path):
    print(f"Error: File '{lisinopril_path}' not found.")
    exit()  # Or handle the error differently

# Print the path for debugging
print(f"Loading model from: {lisinopril_path}")

try:
    with open(lisinopril_path, 'rb') as file:
        lisinopril_model = pickle.load(file)
except FileNotFoundError as e:
    print(f"Error: Failed to open file '{lisinopril_path}'. ({e})")
except pickle.UnpicklingError as e:
    print(f"Error: Error unpickling the model. ({e})")
else:
    # Model loaded successfully
    print("Lisinopril loaded successfully!")


omeprazole_path = "C:/Users/arige/OneDrive/Desktop/major/core/static/Omeprazole.pkl"

# Verify file existence (optional)
if not os.path.exists(omeprazole_path):
    print(f"Error: File '{omeprazole_path}' not found.")
    exit()  # Or handle the error differently

# Print the path for debugging
print(f"Loading model from: {omeprazole_path}")

try:
    with open(omeprazole_path, 'rb') as file:
        omeprazole_model = pickle.load(file)
except FileNotFoundError as e:
    print(f"Error: Failed to open file '{omeprazole_path}'. ({e})")
except pickle.UnpicklingError as e:
    print(f"Error: Error unpickling the model. ({e})")
else:
    # Model loaded successfully
    print("Omeprazole Model loaded successfully!")

def login(request):
    return render(request, 'login.html')

def home(request):
    return render(request, 'home.html')

def verify(request):
    if request.method == 'POST':

        username = request.POST.get('username')
        password = request.POST.get('password')

        print(f"Received username: {username}")
        print(f"Received password: {password}")

        try:
            member = Member.objects.get(name=username)

            print(f"Stored password: {member.password}")
            #this is returning false due to the hashing of password by django , in order to make it work we need to use check_password function and pass the password and the hashed password as parameters
            #print(check_password(password, member.password))

            if password == member.password:
                # Authentication successful

                if member.role == 'Admin':
                    print('Authentication successful')
                    request.session['user_id'] = member.id
                    print('Session id or User id ' , member.id)
                    return render(request, 'admin_dashboard.html', {'member': member})
                print('Authentication successful')
                request.session['user_id'] = member.id
                print('Session id or User id ' , member.id)  # Store user_id in 
                return render(request, 'pharmacy_dashboard.html', {'member': member})
            else:
                # Authentication failed
                print('Authentication failed')
                # show and error page to the user 

                return render(request,'login_error.html', {'error': 'Authentication failed incorrect username or  password'})
        except Member.DoesNotExist:
            # Handle case where username is not found
            print('Username not found')
            return render(request, 'login_error.html', {'error': 'Username not found'})
    

def view(request):
    members = Member.objects.all()
    context = {
        'members': members
    }
    return render(request, 'view.html', context)


def pharmacy_data_entry(request):
    return render(request, 'pharmacy_dataentry.html')

def verify_medicine_pharmacy(request):
    if request.method == 'POST':
        batch_no=request.POST.get('batch_no')
        manufacturer=request.POST.get('manufacturer')
        supplier=request.POST.get('supplier')
        name_of_medicine=request.POST.get('medicine_name')
        expiry= str(request.POST.get('expiry'))
        camount = float(request.POST.get('camount'))
        hamount = float(request.POST.get('hamount')) 
        namount = float(request.POST.get('namount')) 
        oamount = float(request.POST.get('oamount'))  

        print("Received batch_no:", batch_no)
        print("Received manufacturer:", manufacturer)
        print("Received supplier:", supplier)
        print("Received name_of_medicine:", name_of_medicine)
        print("Received expiry_date:", expiry)
        print("Received camount:", camount)
        print("Received hamount:", hamount)
        print("Received namount:", namount)
        print("Received oamount:", oamount)

        # Saving the data to the full medicine list irrespectuve of the test status

        if name_of_medicine == 'Paracetemol':
            prediction = paracetemol_model.predict([[camount, hamount,namount,oamount]])[0]
        elif name_of_medicine == 'Aspirin':
            prediction = aspirin_model.predict([[camount, hamount,oamount]])[0]
        elif name_of_medicine == 'Amlopidine':
            prediction = amlopidine_model.predict([[camount, hamount,oamount,namount]])[0]
        elif name_of_medicine == 'Lisinopril':
            prediction = lisinopril_model.predict([[camount, hamount,oamount,namount]])[0]
        elif name_of_medicine == 'Omeprazole':
            prediction = omeprazole_model.predict([[camount, hamount,oamount,namount]])[0]
        else:
            print('Medicine not found')

        print(prediction)
        if prediction == 1:

            test_status = True
            # Saving the data to the basic database
            medicine_basic_info=Medicine_basic(batch_no=batch_no,manufacturer_name=manufacturer,name_of_medicine=name_of_medicine,expiry_date=expiry,test_status=test_status)
            medicine_basic_info.save()

            return render(request, 'success2.html')
        else:
            test_status = False
            # Saving the data to the basic database
            medicine_basic_info=Medicine_basic(batch_no=batch_no,manufacturer_name=manufacturer,name_of_medicine=name_of_medicine,expiry_date=expiry,test_status=test_status)
            medicine_basic_info.save()
            return render(request, 'test_status_failure.html')
        

def check_test_status(request):
    if request.method == 'POST':
        # Get the batch_no from the form
        batch_no = request.POST.get('batch_no')

        # Query the database for the Medicine instance with the provided batch_no
        try:
            medicine_instance = Medicine_basic.objects.get(batch_no=batch_no)
        except Medicine_basic.DoesNotExist:
            # Handle the case where the batch_no doesn't exist
            return render(request, 'batch_no_not_found.html', {'batch_no': batch_no})

        # Check the test_status
        test_status = medicine_instance.test_status
        print(f"Test status for batch no {batch_no}: {test_status}")
        medicine_name = medicine_instance.name_of_medicine
        print(f"Medicine name for batch no {batch_no}: {medicine_name}")
        expiry_date=medicine_instance.expiry_date
        print(f"Expiry date for batch no {batch_no}: {expiry_date}")
        manufacturer_name=medicine_instance.manufacturer_name
        print(f"Manufacturer name for batch no {batch_no}: {manufacturer_name}")
        # Return the result based on the test_status
        if test_status:
            return render(request, 'test_status_success.html', {'batch_no': batch_no,'medicine_name':medicine_name,'expiry_date':expiry_date,'manufacturer_name':manufacturer_name})
        else:
            return render(request, 'test_status_failure.html', {'batch_no': batch_no})

    return render(request, 'check_test_status.html')

def barcode_upload(request):
    if request.method == 'POST' and request.FILES['barcode_image']:
        barcode_image = request.FILES['barcode_image'].read()

        # Decode the barcode image
        decoded_objects = decode(cv2.imdecode(np.frombuffer(barcode_image, np.uint8), -1))

        if decoded_objects:
            batch_no = decoded_objects[0].data.decode('utf-8')

            print(f"Detected batch no: {batch_no}")


            try:
                medicine_instance = Medicine_basic.objects.get(batch_no=batch_no)
                test_status = medicine_instance.test_status
                print(f"Test status for batch no {batch_no}: {test_status}")
                medicine_name = medicine_instance.name_of_medicine
                print(f"Medicine name for batch no {batch_no}: {medicine_name}")
                expiry_date=medicine_instance.expiry_date
                print(f"Expiry date for batch no {batch_no}: {expiry_date}")
                manufacturer_name=medicine_instance.manufacturer_name
                print(f"Manufacturer name for batch no {batch_no}: {manufacturer_name}")

                if test_status:
                    return render(request, 'test_status_success.html', {'batch_no': batch_no,'medicine_name':medicine_name,'expiry_date':expiry_date,'manufacturer_name':manufacturer_name})
                else:
                    return render(request, 'test_status_failure.html', {'batch_no': batch_no})
            except Medicine_basic.DoesNotExist:
                return render(request, 'batch_no_not_found.html', {'batch_no': batch_no})
        else:
            return render(request, 'barcode_not_detected.html')

    return render(request, 'barcode_upload.html')


def generate_report(request):
    if request.method == 'POST':
        # Collect form data
        data = {
            'issue': request.POST.get('issue'),
            'manufacturer': request.POST.get('manufacturer'),
            'medicine_details': request.POST.get('medicine_details'),
            # Add more fields as needed
        }

        # Generate PDF report
        pdf_content = generate_pdf(data)

        # Save PDF report in a folder
        reports_folder = os.path.join(settings.MEDIA_ROOT, 'reports')
        os.makedirs(reports_folder, exist_ok=True)

        prefix = 'Issue'
        timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
        filename = f"{prefix}_{timestamp}.pdf"
        
        with open(os.path.join(reports_folder, filename), 'wb') as pdf_file:
            pdf_file.write(pdf_content)

        return render(request,'success.html')
    return render(request, 'generate_report.html')

def list_reports(request):
    reports_folder = os.path.join(settings.MEDIA_ROOT, 'reports')
    reports = os.listdir(reports_folder)
    return render(request, 'list_reports.html', {'reports': reports})

def download_report(request, filename):
    reports_folder = os.path.join(settings.MEDIA_ROOT, 'reports')
    file_path = os.path.join(reports_folder, filename)
    with open(file_path, 'rb') as pdf_file:
        response = HttpResponse(pdf_file.read(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    
def generate_pdf(data):
    buffer = io.BytesIO()

    # Create a canvas object
    pdf = Canvas(buffer, pagesize=letter)
    
    # Set up the title
    title = "Report"  # Customize the title as needed

    # Add title to the PDF
    pdf.drawString(100, 750, title)

    # Add data to the PDF
    y_position = 730  # Adjust this value based on your requirements
    for key, value in data.items():
        pdf.drawString(100, y_position, f"{key}: {value}")
        y_position -= 20  # Adjust this value based on your requirements

    # Save the PDF
    pdf.save()

    # Get the PDF content from the buffer
    pdf_content = buffer.getvalue()
    buffer.close()

    return pdf_content
    

def query_results(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]
            results = cursor.fetchall()
        return render(request, 'query_results.html', {'query': query, 'columns': columns, 'results': results})
    return render(request, 'query_form.html')



def fake_medicine_charts(request):
    company_data = Medicine_basic.objects.values('manufacturer_name').annotate(count=Count('id'))
    medicine_data = Medicine_basic.objects.values('name_of_medicine').annotate(count=Count('id'))

    # Process the data
    company_names = [entry['manufacturer_name'] for entry in company_data]
    company_counts = [entry['count'] for entry in company_data]

    medicine_names = [entry['name_of_medicine'] for entry in medicine_data]
    medicine_counts = [entry['count'] for entry in medicine_data]

    # Create bar charts using Plotly
    company_chart = go.Bar(x=company_names, y=company_counts, name='Fake Medicines by Company')
    medicine_chart = go.Bar(x=medicine_names, y=medicine_counts, name='Fake Medicines by Medicine')

    # Render charts
    company_chart_div = go.Figure(data=company_chart, layout={'title': 'Fake Medicines by Company'})
    medicine_chart_div = go.Figure(data=medicine_chart, layout={'title': 'Fake Medicines by Medicine'})

    company_chart_div_html = company_chart_div.to_html(full_html=False)
    medicine_chart_div_html = medicine_chart_div.to_html(full_html=False)

    return render(request, 'fake_medicine_charts.html', {'company_chart': company_chart_div_html, 'medicine_chart': medicine_chart_div_html})


def user_dashboard(request):
    return render(request, 'user_dashboard.html')

def pharmacy_dashboard(request):
    return render(request, 'pharmacy_dashboard.html')

def admin_dashboard(request):
    return render(request, 'admin_dashboard.html')

def success(request):
    return render(request, 'success2.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            uploaded_image = request.FILES['image']
            # Save the uploaded image to a temporary location
            with open('temp_image.jpg', 'wb+') as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)
            # Perform fake detection on the uploaded image
            result = fake_detector('temp_image.jpg')
            # Delete the temporary image file
            os.remove('temp_image.jpg')
            return render(request, 'analysis_result.html', {'result': result})
        except Exception as e:
            # Handle any errors that occur during image processing
            error_message = "An error occurred: {}".format(str(e))
            return render(request, 'error.html', {'error_message': error_message})
    return render(request, 'upload_image.html')

def fake_detector(image_path):
    try:
        # Load the uploaded and real logo images
        uploaded_img = cv2.imread(image_path, 0)  # Grayscale for efficiency
        real_logo_path = os.path.join(settings.MEDIA_ROOT, 'C:/Users/arige/OneDrive/Desktop/ML Major/Logo/real/1.png')  # Use MEDIA_ROOT for logo storage
        real_logo = cv2.imread(real_logo_path, 0)

        if uploaded_img is None or real_logo is None:
            raise Exception("Failed to read one or both images.")

        # Feature extraction using SIFT (consider deep learning for better accuracy)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(uploaded_img, None)
        kp2, des2 = sift.detectAndCompute(real_logo, None)

        # Matching features with FLANN for speed and accuracy balance
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)

        # Ratio test for good matches, consider Lowe's ratio test for robustness
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # Adjust threshold based on dataset
                good_matches.append(m)

        # Determine similarity based on a higher good match threshold for better accuracy
        similarity_threshold = 0.6  # Adjust based on training/validation
        if len(good_matches) >= similarity_threshold * min(len(kp1), len(kp2)):
            return True
        else:
            return False

    except Exception as e:
        raise Exception("Error during fake detection: {}".format(str(e)))