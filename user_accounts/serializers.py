from rest_framework import serializers
from django.core.validators import MinLengthValidator,MaxLengthValidator, MaxValueValidator, MinValueValidator, validate_email
from .models import (CustomUser, Profile, Preference,
                     Religion, UserLike, UploadedImages,
                     Subscription,Community,District
                     ,State,Document, ProfilePicture, CreateSubscription, FamilyName,SuccessStory,ContactDetails,ContactUs)




class CustomUserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[
            MinLengthValidator(8, message="Password must be at least 8 characters long."),
        ]
    )

    confirm_password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[
            MinLengthValidator(8, message="Confirm Password must be at least 8 characters long."),
        ]
    )

    class Meta:
        model = CustomUser
        fields = ("id",
        'first_name','last_name', 'username', 'email', 'password', 'confirm_password', 'profile_for', 'date_of_birth', 'religion', 'community',
        'living_in', 'mobile_number','gender', 'family_name')

        extra_kwargs = {
            'username': {'required': True, 'validators': [MinLengthValidator(4), MaxLengthValidator(150)]},
            'email': {'required': True, 'validators': [validate_email]},
            'profile_for': {'required': True},
            'date_of_birth': {'required': True},
            'religion': {'required': True},
            'community': {'required': True},
            'living_in': {'required': True},
            'gender': {'required': True},
            'mobile_number': {'required': True, 'validators': [MinLengthValidator(10), MaxLengthValidator(15)]},
        }

    def validate(self, data):
        # Check if password and confirm_password match
        if data.get('password') != data.get('confirm_password'):
            raise serializers.ValidationError("Password and confirm_password do not match.")
        return data

    def create(self, validated_data):
        # Remove confirm_password from the validated_data
        validated_data.pop('confirm_password', None)
        # Create a new user with a hashed password
        user = CustomUser.objects.create_user(**validated_data)
        return user

class UserLikeSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserLike
        fields = '__all__'
    display = serializers.BooleanField(default=True)


class UploadedImagesSerializer(serializers.ModelSerializer):
    uploaded_images = serializers.ListField(
        child=serializers.ImageField(
            max_length=1000000,
            allow_empty_file=False,
            use_url=False
        ),
        write_only=True
    )

    class Meta:
        model = UploadedImages
        fields = ['id','user', 'image', 'uploaded_images']

    def create(self, validated_data):
        uploaded_images = validated_data.pop("uploaded_images")
        user = validated_data.get("user")  # Assuming user is passed in validated_data
        uploaded_image_objects = []

        for image in uploaded_images:
            uploaded_image_objects.append(UploadedImages(user=user, image=image))

        # Bulk create UploadedImages objects for better performance
        UploadedImages.objects.bulk_create(uploaded_image_objects)
        return uploaded_image_objects


class ProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Profile
        fields = '__all__'

class PreferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Preference
        fields = '__all__'

class SubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subscription
        fields = '__all__'

class CommunitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Community
        fields = '__all__'


class ReligionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Religion
        fields = '__all__'


class StateSerializer(serializers.ModelSerializer):
    class Meta:
        model = State
        fields = '__all__'

class DistrictSerializer(serializers.ModelSerializer):
    class Meta:
        model = District
        fields = '__all__'

class ProfilePictureSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProfilePicture
        fields = '__all__'

class CreateSubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CreateSubscription
        fields = '__all__'

class ContactUsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ContactUs
        fields = '__all__'

class ContactDetailsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ContactDetails
        fields = '__all__'

class SuccessStorySerializer(serializers.ModelSerializer):
    class Meta:
        model = SuccessStory
        fields = '__all__'


class FamilyNameSerializer(serializers.ModelSerializer):
    class Meta:
        model = FamilyName
        fields = '__all__'


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = '__all__'


class PasswordChangeSerializer(serializers.Serializer):
    username_or_email = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)
