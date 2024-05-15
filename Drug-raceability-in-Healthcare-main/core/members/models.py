from django.db import models

# Create your models here.
class Member(models.Model):
  name = models.CharField(max_length=255)
  role = models.CharField(max_length=255)
  password = models.CharField(max_length=255)
  join_date = models.DateTimeField(auto_now_add=True)

class Medicine_basic(models.Model):
  batch_no = models.CharField(max_length=255)
  manufacturer_name = models.CharField(max_length=255)
  date_of_testing=models.DateTimeField(auto_now_add=True)
  expiry_date = models.DateTimeField()
  name_of_medicine = models.CharField(max_length=255)
  test_status=models.BooleanField(default=False)
