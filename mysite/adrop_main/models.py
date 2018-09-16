from django.db import models

# Create your models here.

class User(models.Model):
    first_name = models.CharField(max_length=30);
    last_name = models.CharField(max_length=30);
    account_name = models.CharField(max_length=30);
    password = models.CharField(max_length=50);
    email_address = models.CharField(max_length=50);
    date_of_birth = models.CharField(max_length=50);
