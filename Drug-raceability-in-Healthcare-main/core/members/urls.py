from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login, name='login'),
    path('verify/', views.verify, name='verify'),
    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
    path('view/', views.view, name='view'),

    path('pharmacy_dashboard/', views.pharmacy_dashboard, name='pharmacy_dashboard'),
    path('pharmacy_data_entry/', views.pharmacy_data_entry, name='pharmacy_data_entry'),
    path('verify_medicine_pharmacy/', views.verify_medicine_pharmacy, name='verify_medicine_pharmacy'),
    path('verify_medicine_pharmacy/success2.html', views.success, name='success'),

    path('check_test_status/', views.check_test_status, name='check_test_status'),
    path('barcode_upload/', views.barcode_upload, name='barcode_upload'),

    path('generate_report/', views.generate_report, name='generate_report'),

    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('list_reports/', views.list_reports, name='list_reports'),
    path('download_report/<str:filename>/', views.download_report, name='download_report'),
    path('query_form/', views.query_results, name='query_results'),
    path('fake_medicine_charts/', views.fake_medicine_charts, name='fake_medicine_charts'),

    path('upload_image/', views.upload_image, name='upload_image'),
]