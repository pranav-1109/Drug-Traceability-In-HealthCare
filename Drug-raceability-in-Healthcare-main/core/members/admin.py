from django.contrib import admin
from .models import Member, Medicine_basic

# Register your models here.

class MemberAdmin(admin.ModelAdmin):
  list_display = ("name", "role", "join_date")

class Medicine_basicAdmin(admin.ModelAdmin):
  list_display = ("batch_no", "manufacturer_name","date_of_testing", "expiry_date", "name_of_medicine", "test_status")

admin.site.register(Member, MemberAdmin)
admin.site.register(Medicine_basic, Medicine_basicAdmin)


