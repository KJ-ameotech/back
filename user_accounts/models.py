from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.crypto import get_random_string

class CustomUser(AbstractUser):
    custom_id = models.CharField(max_length=255, unique=True, editable=False)
    email = models.EmailField(unique=True)
    profile_for =  models.CharField(max_length=255, null=True, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    family_name = models.CharField(max_length=255, null=True, blank=True)
    religion = models.CharField(max_length=255, null=True, blank=True)
    community = models.CharField(max_length=255, null=True, blank=True)
    living_in = models.CharField(max_length=255, null=True, blank=True)
    mobile_number = models.CharField(max_length=15, null=True, blank=True)
    gender = models.CharField(max_length=155, null=True, blank=True)
    latitude = models.CharField(max_length=255, null=True, blank=True)
    longitude = models.CharField(max_length=255, null=True, blank=True)
    approve = models.BooleanField(default=False)

    def save(self, *args, **kwargs):
        if not self.custom_id:
            # Generate a random 7-digit number and prepend 'ID-'
            self.custom_id = f'ID-{get_random_string(length=7, allowed_chars="1234567890")}'

        super(CustomUser, self).save(*args, **kwargs)


class CreateSubscription(models.Model):
    price = models.BigIntegerField()
    subscription_name = models.CharField(max_length=255, unique=True)
    description = models.TextField()
    payment_url = models.CharField(max_length=255, null=True, blank=True)
    def __str__(self):
      return self.subscription_name

class Subscription(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    subscription_type = models.ForeignKey(CreateSubscription, on_delete=models.CASCADE)
    start_date = models.DateField(auto_now_add=True)
    end_date = models.DateField()
    def __str__(self):
      return self.user.username

class UserLike(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='likes_given')
    liked_user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='likes_received')
    timestamp = models.DateTimeField(auto_now_add=True)
    approved = models.BooleanField(default=False)
    slug = models.CharField(max_length=255, null=True, blank=True, unique=True)
    display = models.BooleanField(default=False)
    is_disliked = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.user.username}- request -{self.liked_user.username}'


class Image(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/')

class UploadedImages(models.Model):
    user = models.ForeignKey(CustomUser,on_delete=models.CASCADE,default = "3")
    image  = models.ImageField(upload_to="uploaded_images",default='uploaded_default.jpg')

class Profile(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='profile')
    headline = models.CharField(max_length=255, null=True, blank=True)
    about_me = models.TextField(null=True, blank=True)
    religion = models.CharField(max_length=255, null=True, blank=True)
    caste = models.CharField(max_length=255, null=True, blank=True)
    time_of_birth = models.DateTimeField(null=True, blank=True)
    marital_status = models.CharField(max_length=20, null=True, blank=True)
    height = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    weight = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    education = models.CharField(max_length=255, null=True, blank=True)
    occupation = models.CharField(max_length=255, null=True, blank=True)
    income = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    family_status = models.CharField(max_length=255, null=True, blank=True)
    alcohlic = models.CharField(max_length=255, null=True, blank=True)
    smoker = models.BooleanField(default=False)
    hobbies = models.CharField(max_length=255,null=True, blank=True)
    skin_tone = models.CharField(max_length=155, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username

class Userdata(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    user_image = models.ForeignKey(UploadedImages, on_delete=models.CASCADE)
    user_profile = models.ForeignKey(Profile, on_delete=models.CASCADE)

class ProfilePicture(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, unique=True)
    image = models.ImageField(upload_to="profile_pictures", default="default.jpg")
    approved = models.BooleanField(default=False)

    def __str__(self):
        return self.user.username


class Preference(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    min_age = models.IntegerField(null=True, blank=True)
    max_age = models.IntegerField(null=True, blank=True)
    min_height = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    max_height = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    religion_preference = models.CharField(max_length=255, null=True, blank=True)
    caste_preference = models.CharField(max_length=255, null=True, blank=True)
    marital_status_preference = models.CharField(max_length=20, null=True, blank=True)
    education_preference = models.CharField(max_length=255, null=True, blank=True)
    occupation_preference = models.CharField(max_length=255, null=True, blank=True)
    country_of_residence = models.CharField(max_length=255, null=True, blank=True)
    drinking_habits = models.CharField(max_length=255, null=True, blank=True)
    diet = models.CharField(max_length=255, null=True, blank=True)
    personal_value = models.CharField(max_length=255, null=True, blank=True)
    any_disabblity = models.CharField(max_length=255, null=True, blank=True)
    family_value = models.CharField(max_length=255, null=True, blank=True)
    prefered_state = models.CharField(max_length=255, null=True, blank=True)
    complexion = models.CharField(max_length=255, null=True, blank=True)
    with_children_accpetance = models.CharField(max_length=255, null=True, blank=True)
    smoking_habits = models.CharField(max_length=255, null=True, blank=True)
    body_type = models.CharField(max_length=255, null=True, blank=True)
    manglik = models.CharField(max_length=255, null=True, blank=True)
    mother_tongue = models.CharField(max_length=255, null=True, blank=True)
    prefered_country = models.CharField(max_length=255, null=True, blank=True)
    prefered_status = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username



class Religion(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name


class Community(models.Model):
    religion = models.ForeignKey(Religion, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)


    def __str__(self):
        return self.name

class FamilyName(models.Model):
    community = models.ForeignKey(Community, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name

class State(models.Model):
    name = models.CharField(max_length=100)
    abbreviation = models.CharField(max_length=2, null=True, blank=True)

    def __str__(self):
        return self.name


class District(models.Model):
    name = models.CharField(max_length=100)
    state = models.ForeignKey('State', on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class ContactUs(models.Model):
    name = models.CharField(max_length=255)
    phoneNumber = models.BigIntegerField()
    email = models.EmailField()
    subject = models.CharField(max_length=10000)
    message = models.TextField()

    def __str__(self):
        return self.name


class ContactDetails(models.Model):
    Our_Phone_Number = models.BigIntegerField()
    our_email = models.EmailField()
    timeing = models.CharField(max_length=1000)

class SuccessStory(models.Model):
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255)
    content = models.TextField()
    image = models.ImageField(upload_to='success_story_images/', null=True,blank=True)
    date_published = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.title




class Document(models.Model):
    id_proof = models.ImageField(upload_to='documents/')
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    approved = models.BooleanField(default=False)

    def __str__(self):
        return f"Document for User: {self.user.username}"




