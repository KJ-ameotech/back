from django.contrib.auth import authenticate, login
from rest_framework.exceptions import PermissionDenied
from rest_framework import status, viewsets, permissions
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from django.http import JsonResponse
import numpy as np
from datetime import datetime
from django.http import Http404
from rest_framework import serializers
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.generics import RetrieveAPIView, UpdateAPIView, DestroyAPIView
from .models import *
from .serializers import *
import datetime
from  datetime import date
from django.shortcuts import get_object_or_404
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
import cv2
from datetime import date
from random import randint
from django.views.decorators.csrf import csrf_exempt
import stripe
from django.conf import settings
from geopy.distance import geodesic

import  math

def haversine_distance(lat1=None, lon1=None, lat2=None, lon2=None):
    earth_radius = 6371

    # Convert None values to default values
    if lat1 is None:
        lat1 = 0.0
    if lon1 is None:
        lon1 = 0.0
    if lat2 is None:
        lat2 = 0.0
    if lon2 is None:
        lon2 = 0.0

    # Convert string inputs to floating-point numbers
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c

    return distance


# distance = haversine_distance(lat1, lon1, lat2, lon2)
class CustomUserRegistration(APIView):
    def post(self, request):
        email = request.data.get('email')
        username = request.data.get('username')

        if CustomUser.objects.filter(username=username).exists():
            response_data = {
                'status_code': status.HTTP_400_BAD_REQUEST,
                'error': 'Username already in use.'
            }
            return Response(response_data, status=status.HTTP_400_BAD_REQUEST)

        if CustomUser.objects.filter(email=email).exists():
            response_data = {
                'status_code': status.HTTP_400_BAD_REQUEST,
                'error': 'Email already in use.'
            }
            return Response(response_data, status=status.HTTP_400_BAD_REQUEST)

        serializer = CustomUserSerializer(data=request.data)

        if serializer.is_valid():
            user = serializer.save()

            # Generate JWT tokens using djangorestframework-simplejwt
            refresh = RefreshToken.for_user(user)
            access = str(refresh.access_token)

            response_data = {
                'status_code': status.HTTP_201_CREATED,
                'message': 'Welcome, your account is successfully registered.',
                'user_id': user.id,
                'access_token': access,
                'refresh_token': str(refresh)
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        else:
            response_data = {
                'status_code': status.HTTP_400_BAD_REQUEST,
                'errors': serializer.errors
            }
            return Response(response_data, status=status.HTTP_400_BAD_REQUEST)



class CustomUserDetailView(RetrieveAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = CustomUserSerializer
    lookup_field = 'id'
    def get(self, request, *args, **kwargs):
	    # Get the object to be serialized
	    instance = self.get_object()

	    # Exclude the 'password' field from the serialized representation
	    serializer = self.get_serializer(instance) 
	    data = serializer.data
	    data.pop('password', None)

	    return Response(data)  # or 'pk' depending on how you want to identify users
	 

class CustomUserUpdateView(APIView):
    def patch(self, request, id):
        try:
            user = CustomUser.objects.get(id=id)
        except CustomUser.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

        serializer = CustomUserSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



class CustomUserDeleteView(DestroyAPIView):
    queryset = CustomUser.objects.all()
    lookup_field = 'id'  # or 'pk' depending on how you want to identify users


class UserLikeListCreateView(generics.ListCreateAPIView):
    queryset = UserLike.objects.all()
    serializer_class = UserLikeSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Check if 'user' and 'liked_user' exist in the database
        user_id = serializer.validated_data['user'].id
        liked_user_id = serializer.validated_data['liked_user'].id

        user_exists = CustomUser.objects.filter(id=user_id).exists()
        liked_user_exists = CustomUser.objects.filter(id=liked_user_id).exists()

        if not user_exists or not liked_user_exists:
            return Response({'error': 'User or liked user does not exist.'}, status=status.HTTP_400_BAD_REQUEST)

        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class UserLikeDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = UserLike.objects.all()
    serializer_class = UserLikeSerializer
    lookup_field = 'pk'


    def get(self, request, *args, **kwargs):
        try:
            user_like = self.get_object()
            serializer = self.get_serializer(user_like)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except UserLike.DoesNotExist:
            return Response({'error': 'UserLike does not exist.'}, status=status.HTTP_404_NOT_FOUND)
    


class UserLogin(APIView):
    def post(self, request):
        email = request.data.get('email')
        username = request.data.get('username')
        password = request.data.get('password')
        latitude = request.data.get('latitude')
        longitude = request.data.get('longitude')
        request.data.pop('latitude', None)
        request.data.pop('longitude', None)

        if email is None and username is None:
            # If both email and username are missing, return an error
            response_data = {
                'status_code': status.HTTP_400_BAD_REQUEST,
                'detail': 'Please provide either username or email.',
            }
            return Response(response_data, status=status.HTTP_400_BAD_REQUEST)

        if email is not None:
            try:
                user = CustomUser.objects.get(email=email)
                
            except CustomUser.DoesNotExist:
                user = None
            try:
                userdata = CustomUser.objects.get(email=email)
                user_id = userdata.id
                username = userdata.username
            except:
                response_data = {
                    'status_code': status.HTTP_404_NOT_FOUND,
                    'detail': 'Invalid username or password',
                }
                return Response(response_data, status=status.HTTP_404_NOT_FOUND)
            
            if user is not None:
                # Check if the account status is activ
                    # Authenticate the user using email
                    if user.check_password(password):
                        # Password matches, log the user in
                        if user.approve:

                            login(request, user)
                            user.latitude = latitude
                            user.longitude = longitude
                            user.save()

                            # Generate refresh and access tokens
                            refresh = RefreshToken.for_user(user)
                            response_data = {
                                'status_code': status.HTTP_200_OK,
                                'access_token': str(refresh.access_token),
                                'refresh_token': str(refresh),
                                'user_id': user_id,
                                'username': username
                            }
                            return Response(response_data, status=status.HTTP_200_OK)
                        else:
                            response_data = {
                                'status_code': status.HTTP_403_FORBIDDEN,
                                'detail': "You are under Admin's verification. You will be able to login after 24 Hours",
                            }
                            return Response(response_data, status=status.HTTP_403_FORBIDDEN)

                    else:
                        # If authentication fails, return an error response
                        response_data = {
                            'status_code': status.HTTP_401_UNAUTHORIZED,
                            'detail': 'Invalid credentials',
                        }
                        return Response(response_data, status=status.HTTP_401_UNAUTHORIZED)

            else:
                # If user not found, return an error response
                response_data = {
                    'status_code': status.HTTP_404_NOT_FOUND,
                    'detail': 'Invalid username or password',
                }
                return Response(response_data, status=status.HTTP_404_NOT_FOUND)
        
        else:
            # Authenticate the user
            user = authenticate(request, username=username, password=password)
            try:
                userdata = CustomUser.objects.get(username=username)
                user_id = userdata.id
            except:
                response_data = {
                    'status_code': status.HTTP_404_NOT_FOUND,
                    'detail': 'Invalid username or password',
                }
                return Response(response_data, status=status.HTTP_404_NOT_FOUND)


            if user is not None:
                # Check if the account status is active
                if user.approve:
                    # If authentication is successful, log the user in
                    login(request, user)

                    user.latitude = latitude
                    user.longitude = longitude
                    user.save()

                    # Generate refresh and access tokens
                    refresh = RefreshToken.for_user(user)
                    response_data = {
                        'status_code': status.HTTP_200_OK,
                        'access_token': str(refresh.access_token),
                        'refresh_token': str(refresh),
                        'user_id': user_id,
                        'username': username
                    }
                    return Response(response_data, status=status.HTTP_200_OK)
                else:
                    # If account status is not active, return an error response
                    response_data = {
                        'status_code': status.HTTP_403_FORBIDDEN,
                        'detail': "You are under Admin's verification. You will be able to login after 24 Hours",
                    }
                    return Response(response_data, status=status.HTTP_403_FORBIDDEN)
            else:
                # If authentication fails, return an error response
                response_data = {
                    'status_code': status.HTTP_401_UNAUTHORIZED,
                    'detail': 'Invalid credentials',
                }
                return Response(response_data, status=status.HTTP_401_UNAUTHORIZED)



class ProfileListCreateView(APIView):
    def get(self, request):

        user_id = request.query_params.get('user_id')

        liked_users = UserLike.objects.filter(user=user_id)

        liked_user_ids = set([liked.liked_user_id for liked in liked_users])

        queryset = CustomUser.objects.exclude(id=user_id).exclude(id__in=liked_user_ids)

        user_ids = [user.id for user in queryset]

        if not user_id:
            return Response({'detail': 'user_id query parameter is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            requesting_user = Profile.objects.get(user_id=user_id)
        except Profile.DoesNotExist:
            return Response({'detail': 'User with the provided user_id does not exist.'},
                            status=status.HTTP_404_NOT_FOUND)


        profiles = Profile.objects.filter(user__in=user_ids)
        profile_data = []
        latitude = CustomUser.objects.get(id=user_id)

        requesting_user_latitude = latitude.latitude  # Example latitude within [-90, 90]
        requesting_user_longitude = latitude.longitude

        for profile in profiles:
            try:
                profile_picture = ProfilePicture.objects.get(user=profile.user)
                user_latitude = profile.user.latitude
                user_longitude = profile.user.longitude

                distance = round(geodesic((requesting_user_latitude, requesting_user_longitude),
                                          (user_latitude, user_longitude)).kilometers)
                if distance is None:
                    distance = "Too far"
                today = date.today()
                date_of_birth = profile.user.date_of_birth
                if date_of_birth:
                    age = today.year - date_of_birth.year - (
                                (today.month, today.day) < (date_of_birth.month, date_of_birth.day))
                else:
                    age = None
                profile_data.append({
                    'user_id': profile.user.id,
                    'first_name': profile.user.first_name,
                    'last_name': profile.user.last_name,
                    'username': profile.user.username,
                    'age': age,
                    'email': profile.user.email,
                    'profile_picture': profile_picture.image.url,
                    'distance': distance
                })
            except ProfilePicture.DoesNotExist:
                profile_data.append({
                    'user_id': profile.user.id,
                    'first_name': profile.user.first_name,
                    'last_name': profile.user.last_name,
                    'username': profile.user.username,
                    'email': profile.user.email,
                    'profile_picture': None,
                    'distance': "Too far"
                })

        return Response(profile_data)

    def post(self, request):
        user = request.data.get('user')

        existing_profile = Profile.objects.filter(user=user).first()
        if existing_profile:
            return Response({'detail': 'Profile already exists for this user.'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = ProfileSerializer(data=request.data)

        if serializer.is_valid():
            profile = serializer.save()

            # If a profile picture is included in the request, create it
            profile_picture_data = request.data.get('profile_picture')
            if profile_picture_data:
                profile_picture_serializer = ProfilePictureSerializer(
                    data={'user': profile.id, 'image': profile_picture_data}
                )
                if profile_picture_serializer.is_valid():
                    profile_picture_serializer.save()
                else:
                    # Handle errors in the profile picture creation
                    return Response(
                        profile_picture_serializer.errors,
                        status=status.HTTP_400_BAD_REQUEST
                    )

            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




class ProfileRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Profile.objects.all()
    serializer_class = ProfileSerializer
    lookup_field = 'user'

    def update(self, request, *args, **kwargs):
        # Use partial update for PATCH
        partial = True
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        return Response(serializer.data)




class UploadedImagesListCreateView(generics.ListCreateAPIView):
    queryset = UploadedImages.objects.all()
    serializer_class = UploadedImagesSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Save the uploaded images and get the created instances
        uploaded_images = serializer.save()

        # Construct a custom response with details about the created images
        response_data = {
            "message": "Images uploaded successfully.",
            "images": UploadedImagesSerializer(uploaded_images, many=True).data
        }

        return Response(response_data, status=status.HTTP_201_CREATED)

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        instance.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)




class PreferenceListCreateView(generics.ListCreateAPIView):
    queryset = Preference.objects.all()
    serializer_class = PreferenceSerializer


class PreferenceDetailView(APIView):
    def get(self, request):
        user_id = request.query_params.get('user_id')
        preferences = Preference.objects.filter(user=user_id)
        serializer = PreferenceSerializer(preferences, many=True)
        return Response(serializer.data)

    def patch(self, request):
        user_id = request.query_params.get('user_id')
        preferences = Preference.objects.filter(user=user_id).first()
        if not preferences:
            return Response(status=status.HTTP_404_NOT_FOUND)

        serializer = PreferenceSerializer(preferences, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        user_id = request.query_params.get('user_id')
        preferences = Preference.objects.filter(user=user_id).first()
        if not preferences:
            return Response(status=status.HTTP_404_NOT_FOUND)

        preferences.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class SubscriptionListAPIView(APIView):
    def get(self, request):
        subscriptions = Subscription.objects.all()
        serializer = SubscriptionSerializer(subscriptions, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = SubscriptionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SubscriptionDetailAPIView(APIView):
    def get_object(self, pk):
        try:
            return Subscription.objects.get(pk=pk)
        except Subscription.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        subscription = self.get_object(pk)
        serializer = SubscriptionSerializer(subscription)
        return Response(serializer.data)

    def put(self, request, pk):
        subscription = self.get_object(pk)
        serializer = SubscriptionSerializer(subscription, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        subscription = self.get_object(pk)
        subscription.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class CommunityList(APIView):
    def get(self, request):
        communities = Community.objects.all()
        serializer = CommunitySerializer(communities, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = CommunitySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



class CommunityDetail(APIView):
    def get_object(self, pk):
        try:
            return Community.objects.get(pk=pk)
        except Community.DoesNotExist:
            return None

    def get(self, request, pk):
        community = self.get_object(pk)
        if community is not None:
            serializer = CommunitySerializer(community)
            return Response(serializer.data)
        return Response({'detail': 'Community not found'}, status=status.HTTP_404_NOT_FOUND)

    def put(self, request, pk):
        community = self.get_object(pk)
        if community is not None:
            serializer = CommunitySerializer(community, data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response({'detail': 'Community not found'}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, pk):
        community = self.get_object(pk)
        if community is not None:
            community.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response({'detail': 'Community not found'}, status=status.HTTP_404_NOT_FOUND)



class ReligionList(APIView):
    def get(self, request):
        religions = Religion.objects.all()
        serializer = ReligionSerializer(religions, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = ReligionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ReligionDetail(APIView):
    def get_object(self, pk):
        try:
            return Religion.objects.get(pk=pk)
        except Religion.DoesNotExist:
            return None

    def get(self, request, pk):
        religion = self.get_object(pk)
        if religion is not None:
            serializer = ReligionSerializer(religion)
            return Response(serializer.data)
        return Response({'detail': 'Religion not found'}, status=status.HTTP_404_NOT_FOUND)

    def put(self, request, pk):
        religion = self.get_object(pk)
        if religion is not None:
            serializer = ReligionSerializer(religion, data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response({'detail': 'Religion not found'}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, pk):
        religion = self.get_object(pk)
        if religion is not None:
            religion.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response({'detail': 'Religion not found'}, status=status.HTTP_404_NOT_FOUND)

class CommunityByReligionBy(APIView):
    def get(self, request, religion_id):
        try:
            # Filter religions by religion's ID
            religion = Religion.objects.get(id=religion_id)
            communities = Community.objects.filter(religion=religion)
            serializer = ReligionSerializer(communities, many=True)  # Use 'communities' here, not 'religions'
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Religion.DoesNotExist:
            return Response({'detail': 'Religion not found'}, status=status.HTTP_404_NOT_FOUND)


class StateList(APIView):
    def get(self, request):
        states = State.objects.all()
        serializer = StateSerializer(states, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = StateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class StateDetail(APIView):
    def get_object(self, pk):
        try:
            return State.objects.get(pk=pk)
        except State.DoesNotExist:
            return None

    def get(self, request, pk):
        state = self.get_object(pk)
        if state is not None:
            serializer = StateSerializer(state)
            return Response(serializer.data)
        return Response({'detail': 'State not found'}, status=status.HTTP_404_NOT_FOUND)

    def put(self, request, pk):
        state = self.get_object(pk)
        if state is not None:
            serializer = StateSerializer(state, data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response({'detail': 'State not found'}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, pk):
        state = self.get_object(pk)
        if state is not None:
            state.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response({'detail': 'State not found'}, status=status.HTTP_404_NOT_FOUND)


class DistrictList(APIView):
    def get(self, request):
        districts = District.objects.all()
        serializer = DistrictSerializer(districts, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = DistrictSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DistrictDetail(APIView):
    def get_object(self, pk):
        try:
            return District.objects.get(pk=pk)
        except District.DoesNotExist:
            return None

    def get(self, request, pk):
        district = self.get_object(pk)
        if district is not None:
            serializer = DistrictSerializer(district)
            return Response(serializer.data)
        return Response({'detail': 'District not found'}, status=status.HTTP_404_NOT_FOUND)

    def put(self, request, pk):
        district = self.get_object(pk)
        if district is not None:
            serializer = DistrictSerializer(district, data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response({'detail': 'District not found'}, status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, pk):
        district = self.get_object(pk)
        if district is not None:
            district.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response({'detail': 'District not found'}, status=status.HTTP_404_NOT_FOUND)


class DistrictsByState(APIView):
    def get(self, request, state_id):
        try:
            # Perform a lookup by state ID
            state = State.objects.get(id=state_id)

            # Filter districts by the retrieved state
            districts = District.objects.filter(state=state)
            serializer = DistrictSerializer(districts, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except State.DoesNotExist:
            return Response({'detail': 'State not found'}, status=status.HTTP_404_NOT_FOUND)


class UserProfilepictureListCreateView(generics.ListCreateAPIView):
    queryset = ProfilePicture.objects.all()
    serializer_class = ProfilePictureSerializer
    parser_classes = (MultiPartParser,)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        uploaded_image = serializer.validated_data['image']

        nparr = np.frombuffer(uploaded_image.read(), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = detect_faces_in_image(img_np)

        if result:
            self.perform_create(serializer)
            return Response({"detail": "Single face detected. Profile picture created."},
                            status=status.HTTP_201_CREATED)
        else:
            return Response({"detail": "Multiple faces detected or photo doesn't contain a human face"},
                            status=status.HTTP_400_BAD_REQUEST)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def detect_faces_in_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 1:
        print("Single face detected.")
        return True
    elif len(faces) > 1:
        print("Multiple faces detected.")
        return False
    else:
        print("No human face detected.")
        return False


class UserProfilepictureDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = ProfilePicture.objects.all()
    serializer_class = ProfilePictureSerializer
    lookup_field = 'user_id'
    parser_classes = (MultiPartParser,)

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        data = serializer.data
        data['status_code'] = status.HTTP_200_OK 
        return Response(data)


    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data)
        
        if serializer.is_valid():
            uploaded_image = request.data.get('image')

            if uploaded_image:
                nparr = np.frombuffer(uploaded_image.read(), np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Detect faces in the image
                result = detect_faces_in_image(img_np)

                if not result:
                    return Response(
                        {"detail": "Multiple faces detected or photo doesn't contain a human face"},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            serializer.save()
            data = serializer.data
            data['status_code'] = status.HTTP_200_OK 
            return Response(data)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        instance.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class CheckEmailExists(APIView):
	def post(self, request):
	        email = request.data.get('email')
	        phone_number = request.data.get('phone_number')

	        email_exists = CustomUser.objects.filter(email=email).exists()
	        phone_number_exists = CustomUser.objects.filter(mobile_number=phone_number).exists()

	        if email_exists and phone_number_exists:
	            return Response({'message': 'Email and phone number both exist', 'status': status.HTTP_200_OK}, status=status.HTTP_200_OK)
	        else:
	            return Response({'exists': False, 'status': status.HTTP_404_NOT_FOUND}, status=status.HTTP_404_NOT_FOUND)



class ChangePassword(APIView):
    def post(self, request):
        email = request.data.get('email')
        new_password = request.data.get('new_password')

        User = get_user_model()

        try:
            user = User.objects.get(email=email)
            user.set_password(new_password)  # Set and hash the password
            user.save()  # Save the user with the hashed password

            return Response({'message': 'Password Changed successfully'}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)





from django.utils import timezone
from django.db.models import Q



class CustomUserSearchAPIView(APIView):
    def get(self, request):
        current_user_id = self.request.query_params.get('user_id')

        if current_user_id is None:
            return Response({'detail': 'user_id query parameter is required.'}, status=status.HTTP_400_BAD_REQUEST)

        # CustomUser search parameters
        age_from = self.request.query_params.get('age_from')
        age_to = self.request.query_params.get('age_to')
        gender = self.request.query_params.get('gender')
        community = self.request.query_params.get('community')

        user_id_excluded = []

        user_id_excluded.append(current_user_id)

        user_profiles = UserLike.objects.filter((Q(liked_user_id=current_user_id) | Q(user=current_user_id)))

        for data in user_profiles:
            if data.user.id != int(current_user_id):
                user_id_excluded.append(data.user.id)
            else:
                user_id_excluded.append(data.liked_user.id)

        print(user_id_excluded)

        # disliked_users = UserLike.objects.filter((Q(is_disliked=True) & Q(liked_user_id=current_user_id)))
        # disliked_user_ids = set([disliked.liked_user_id for disliked in disliked_users])
        # liked_users = UserLike.objects.filter(user=current_user_id)
        # liked_user_ids = set([liked.liked_user_id for liked in liked_users])

        # CustomUser search
        # custom_user_queryset = CustomUser.objects.exclude(id=current_user_id).exclude(id__in=liked_user_ids).exclude(id__in=disliked_user_ids)
        custom_user_queryset = CustomUser.objects.exclude(id__in=user_id_excluded)
        profileuser = Profile.objects.exclude(user__in=user_id_excluded)
        # if age_from is not None and age_to is not None:
        #     try:
        #         currentYear = date.today().year
        #         age_from_year = int(age_from)
        #         age_to_year = int(age_to)
        #         age_from_date = date(currentYear - age_to_year, 1, 1)
        #         age_to_date = date(currentYear - age_from_year, 12, 31)
        #         custom_user_queryset = profileuser.filter(time_of_birth__date__range=(age_from_date, age_to_date))
        #         print(custom_user_queryset, "=============================")
        #     except ValueError:
        #         return Response({'detail': 'Invalid year format. Year must be an integer.'},
        #                         status=status.HTTP_400_BAD_REQUEST)

        if gender:
            custom_user_queryset = custom_user_queryset.filter(gender=gender)
            print(custom_user_queryset)

        if community:
            custom_user_queryset = custom_user_queryset.filter(family_name=community)
            print(custom_user_queryset)

        custom_user_serializer = CustomUserSerializer(custom_user_queryset, many=True)
        custom_user_serialized_data = custom_user_serializer.data

        basic_user_data = []

        # Check if there are any query parameters for the Profile model
        profile_query_params_exist = any(
            param in request.GET for param in
            ['startheight', 'endheight', 'education', 'marital_status', 'minweight', 'maxweight', 'minincome', 'maxincome','skin_tone']
        )

        if profile_query_params_exist:
            # Profile search parameters
            start_height = self.request.query_params.get('startheight')
            end_height = self.request.query_params.get('endheight')
            education = self.request.query_params.get('education')
            marital_status = self.request.query_params.get('marital_status')
            min_weight = self.request.query_params.get('minweight')
            max_weight = self.request.query_params.get('maxweight')
            min_income = self.request.query_params.get('minincome')
            max_income = self.request.query_params.get('maxincome')
            skin_tone = self.request.query_params.get('skin_tone')

            # Additional filtering of Profile using results from CustomUser
            profile_queryset = Profile.objects.filter(user__in=custom_user_queryset)

            print(profile_queryset)

            if education is not None:
                profile_queryset = profile_queryset.filter(education=education)


            if marital_status is not None:
                profile_queryset = profile_queryset.filter(marital_status=marital_status)

            if start_height is not None and end_height is not None:
                profile_queryset = profile_queryset.filter(height__range=(start_height, end_height))

            if min_weight is not None and max_weight is not None:
                profile_queryset = profile_queryset.filter(weight__range=(min_weight, max_weight))

            if min_income is not None and max_income is not None:
                profile_queryset = profile_queryset.filter(income__range=(min_income, max_income))

            if skin_tone is not None:
                profile_queryset = profile_queryset.filter(skin_tone=skin_tone)

            profile_serializer = ProfileSerializer(profile_queryset, many=True)
            profile_serialized_data = profile_serializer.data

            combined_profile_data = []

            latitude = CustomUser.objects.get(id=current_user_id)
            requesting_user_latitude = latitude.latitude
            requesting_user_longitude = latitude.longitude

            for profile in profile_serialized_data:
                user = profile['user']
                try:
                    dob = Profile.objects.get(user=user)
                    dob = dob.time_of_birth.date()
                    current_date = date.today()
                    age = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))
                except:
                    age = None

                try:
                    profile_picture = ProfilePicture.objects.get(user=user)
                except ProfilePicture.DoesNotExist:
                    profile_picture = None

                user_location = CustomUser.objects.filter(id=user).first()

                user_latitude = user_location.latitude
                user_longitude = user_location.longitude

                # # Calculate the distance using geopy.distance
                distance = round(geodesic((requesting_user_latitude, requesting_user_longitude),
                                          (user_latitude, user_longitude)).kilometers)

                if distance is None:
                    distance = "Too far"

                today = date.today()
                date_of_birth = user_location.date_of_birth


                combined_profile_data.append({
                    'user_id': user_location.id,
                    'custom_id':user_location.custom_id,
                    'first_name': user_location.first_name,
                    'last_name': user_location.last_name,
                    'username': user_location.username,
                    'age': age,
                    'email': user_location.email,
                    'profile_picture': profile_picture.image.url if profile_picture.approved else None,
                    'distance': distance
                    # Add other profile data fields here
                })

            return Response(combined_profile_data, status=status.HTTP_200_OK)
        for users in custom_user_serialized_data:
            user = users['id']
            try:
                dob = Profile.objects.get(user=user)
                dob =  dob.time_of_birth.date()
                current_date = date.today()
                age = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))
            except:
                age = None
            try:
                profile_picture = ProfilePicture.objects.get(user=user)
            except ProfilePicture.DoesNotExist:
                profile_picture = None

            user_location = CustomUser.objects.filter(id=user).first()

            user_latitude = user_location.latitude
            user_longitude = user_location.longitude

            current_user_data = CustomUser.objects.filter(id=current_user_id).first()
            current_user_latitude = current_user_data.latitude
            current_user_longitude = current_user_data.longitude

            # # Calculate the distance using geopy.distance
            distance = round(geodesic((current_user_latitude, current_user_longitude),
                                      (user_latitude, user_longitude)).kilometers)

            if distance is None:
                distance = "Too far"

            today = date.today()
            date_of_birth = user_location.date_of_birth


            basic_user_data.append({
                'user_id': user_location.id,
                'first_name': user_location.first_name,
                'custom_id': user_location.custom_id,
                'last_name': user_location.last_name,
                'username': user_location.username,
                'age': age,
                'email': user_location.email,
                'profile_picture': profile_picture.image.url if profile_picture.approved else None,
                'distance': distance
                # Add other profile data fields here
            })
        return Response(basic_user_data, status=status.HTTP_200_OK)

class UserLikeAPIView(APIView):
    serializer_class = UserLikeSerializer  # Replace with your actual serializer

    def get(self, request, *args, **kwargs):
        # Get the liked_user_id from the URL query parameter
        liked_user_id = self.request.query_params.get('liked_user_id')

        # Check if liked_user_id is provided in the query parameters
        if liked_user_id is not None:
            # Filter UserLike objects based on the liked_user_id
            queryset = UserLike.objects.filter(Q(liked_user_id=liked_user_id) & Q(is_disliked = False))
            serializer = self.serializer_class(queryset, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            # If liked_user_id is not provided, return a 400 Bad Request response
            return Response({"error": "liked_user_id parameter is required"}, status=status.HTTP_400_BAD_REQUEST)

class UserLikeCountAPIView(APIView):
    serializer_class = UserLikeSerializer  # Replace with your actual serializer

    def get(self, request, *args, **kwargs):
        # Get the liked_user_id from the URL query parameter
        liked_user_id = self.request.query_params.get('liked_user_id')
        # Check if liked_user_id is provided in the query parameters
        if liked_user_id is not None:
            # Filter UserLike objects based on the liked_user_id and get the count
            count = UserLike.objects.filter(liked_user_id=liked_user_id).count()
            return Response({"count": count}, status=status.HTTP_200_OK)
        else:
            # If liked_user_id is not provided, return a 400 Bad Request response
            return Response({"error": "liked_user_id parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
class CustomUserList(APIView):
    def get(self, request):
        users = CustomUser.objects.all()
        user_serializer = CustomUserSerializer(users, many=True)

        user_data_with_images_and_profile = []

        for user in user_serializer.data:
            user_data = user  # Copy the user data to a new dictionary
            # Get the user's profile picture if available
            try:
                profile_picture = ProfilePicture.objects.get(user=user['id'])
                profile_picture_serializer = ProfilePictureSerializer(profile_picture)
                user_data['profile_picture'] = profile_picture_serializer.data
            except ProfilePicture.DoesNotExist:
                user_data['profile_picture'] = None  # No profile picture found
            # Get the user's profile data if available
            try:
                profile_data = Profile.objects.get(user=user['id'])  # Assuming UserProfile is your profile model
                profile_data_serializer = ProfileSerializer(profile_data)  # Create a serializer for UserProfile
                user_data['profile_data'] = profile_data_serializer.data
            except Profile.DoesNotExist:
                user_data['profile_data'] = None  # No profile data found
            user_data_with_images_and_profile.append(user_data)
        return Response(user_data_with_images_and_profile)


class UserLikeListViewRequestsAccepted(APIView):
    def get(self, request, user_id):
        user_likes = UserLike.objects.filter(user_id=user_id, approved=True)
        liked_user_ids = [user_like.liked_user.id for user_like in user_likes]
        user_data = CustomUser.objects.filter(id=user_id)
        liked_users_data = CustomUser.objects.filter(id__in=liked_user_ids)
        profile_data = Profile.objects.filter(user_id__in=[user_id] + liked_user_ids)
        profile_picture_data = ProfilePicture.objects.filter(user_id__in=[user_id] + liked_user_ids)
        user_likes_data = UserLikeSerializer(user_likes, many=True).data
        user_data = CustomUserSerializer(user_data, many=True).data
        liked_users_data = CustomUserSerializer(liked_users_data, many=True).data
        profile_data = ProfileSerializer(profile_data, many=True).data
        profile_picture_data = ProfilePictureSerializer(profile_picture_data, many=True).data
        user_data_dict = {user['id']: user for user in user_data}
        liked_users_data_dict = {user['id']: user for user in liked_users_data}
        # profile_data_dict = {profile['user']: profile for profile in profile_data}
        profile_picture_data_dict = {picture['user']: picture for picture in profile_picture_data}

        merged_data = []

        for user_like in user_likes_data:
            user_id = user_like['user']
            liked_user_id = user_like['liked_user']
            user_data_entry = user_data_dict.get(user_id)
            liked_user_data_entry = liked_users_data_dict.get(liked_user_id)
            # profile_data_entry = profile_data_dict.get(user_id)
            user_profile_picture_data_entry = profile_picture_data_dict.get(user_id)
            liked_user_profile_picture_data_entry = profile_picture_data_dict.get(liked_user_id)

            if user_data_entry:
                merged_entry = {**user_like, 'user_data': user_data_entry}

                if liked_user_data_entry:
                    merged_entry['liked_user_data'] = liked_user_data_entry

                # if profile_data_entry:
                #     merged_entry['profile_data'] = profile_data_entry

                if user_profile_picture_data_entry:
                    merged_entry['user_profile_picture_data'] = user_profile_picture_data_entry

                if liked_user_profile_picture_data_entry:
                    merged_entry['liked_user_profile_picture_data'] = liked_user_profile_picture_data_entry

                merged_data.append(merged_entry)

        return Response(merged_data, status=status.HTTP_200_OK)



class LikedUserLikeListViewRequestsAccepted(APIView):
    def get(self, request, liked_user_id):
        # Retrieve UserLike objects where the specified user is liked (liked_user) with approved=True
        user_likes = UserLike.objects.filter(liked_user_id=liked_user_id, approved=True)
        liked_user = UserLike.objects.filter(user=liked_user_id, approved=True)

        # Extract user IDs who liked the specified user
        user_ids = [user_like.user.id for user_like in user_likes]
        like_user_ids = [c_user.liked_user.id for c_user in liked_user]

        user_ids = user_ids + like_user_ids

        # Check if the user has an active subscription
        user_has_active_subscription, active_subscription = self.check_user_has_active_subscription(liked_user_id)

        user_data = CustomUser.objects.filter(id__in=user_ids)

        user_data = CustomUserSerializer(user_data, many=True).data
        # for index, user in enumerate(user_data):
        #     serialized_data[index]['custom_id'] = user.custom_id

        profile_images = ProfilePicture.objects.filter(user__id__in=user_ids)
        profile_images_data = ProfilePictureSerializer(profile_images, many=True).data

        response_data = {
            "user_data": user_data,
            "profile_images_data": profile_images_data,
            "subscription_name": active_subscription.subscription_type.subscription_name if user_has_active_subscription else None,
        }

        for user, profile_image_data in zip(response_data["user_data"], response_data["profile_images_data"]):
            slugList = user_likes.filter(user=user['id'])
            if not slugList:
                slugList = liked_user.filter(liked_user=user['id'])

            user['profile_image'] = profile_image_data['image']
            user['slug'] = slugList[0].slug

        return Response(response_data, status=status.HTTP_200_OK)

    def check_user_has_active_subscription(self, user_id):
        # Get the current user
        user = get_object_or_404(CustomUser, id=user_id)

        # Check if the user has an active subscription
        active_subscription = Subscription.objects.filter(user=user, end_date__gte=timezone.now()).first()

        return active_subscription is not None, active_subscription

class ProfileSearchView(APIView):
    serializer_class = ProfileSerializer
    def get(self, request):
        current_user_id = self.request.query_params.get('user_id')
        if not current_user_id:
            return Response({'detail': 'user_id query parameter is required.'}, status=status.HTTP_400_BAD_REQUEST)

        query_params = {}  # Initialize an empty dictionary for query parameters

        start_height = self.request.query_params.get('startheight')
        end_height = self.request.query_params.get('endheight')
        caste = self.request.query_params.get('caste')
        marital_status = self.request.query_params.get('marital_status')
        min_weight = self.request.query_params.get('minweight')
        max_weight = self.request.query_params.get('maxweight')
        min_income = self.request.query_params.get('minincome')
        max_income = self.request.query_params.get('maxincome')
        skin_tone = self.request.query_params.get('skin_tone')

        liked_users = UserLike.objects.filter(user=current_user_id)
        liked_user_ids = set([liked.liked_user_id for liked in liked_users])

        queryset = Profile.objects.exclude(user=current_user_id).exclude(user__in=liked_user_ids)


        if caste is not None:
            queryset = queryset.filter(caste=caste)

        if marital_status is not None:
            queryset = queryset.filter(marital_status=marital_status)

        if start_height and end_height is not None:
            queryset = queryset.filter(height__range=(start_height, end_height))

        if min_weight and max_weight is not None:
            queryset = queryset.filter(weight__range=(min_weight, max_weight))

        if min_income and max_income is not None:
            queryset = queryset.filter(income__range=(min_income, max_income))

        if skin_tone is not None:
            queryset = queryset.filter(skin_tone=skin_tone)

        serializer = ProfileSerializer(queryset, many=True)
        serialized_data = serializer.data

        # Extract the 'id' field from each user in the serialized data
        user_ids = [user['id'] for user in serialized_data]

        profiles = Profile.objects.filter(user__in=user_ids)
        profiles = profiles.exclude(user=current_user_id)
        profile_data = []
        latitude = CustomUser.objects.get(id=current_user_id)

        requesting_user_latitude = latitude.latitude
        requesting_user_longitude = latitude.longitude

        for profile in profiles:
            try:
                profile_picture = ProfilePicture.objects.get(user=profile.user)
                user_latitude = profile.user.latitude
                user_longitude = profile.user.longitude

                # Calculate the distance using geopy.distance
                distance = round(geodesic((requesting_user_latitude, requesting_user_longitude),
                                          (user_latitude, user_longitude)).kilometers)
                if distance is None:
                    distance = "Too far"
                today = date.today()
                date_of_birth = profile.user.date_of_birth
                if date_of_birth:
                    age = today.year - date_of_birth.year - (
                            (today.month, today.day) < (date_of_birth.month, date_of_birth.day))
                else:
                    age = None
                profile_data.append({
                    'user_id': profile.user.id,
                    'first_name': profile.user.first_name,
                    'last_name': profile.user.last_name,
                    'username': profile.user.username,
                    'age': age,
                    'email': profile.user.email,
                    'profile_picture': profile_picture.image.url if profile_picture and profile_picture.approved  else None,
                    'distance': distance
                    # Add other profile data fields here
                })
            except ProfilePicture.DoesNotExist:
                # Handle the case where a user doesn't have a profile picture
                profile_data.append({
                    'user_id': profile.user.id,
                    'first_name': profile.user.first_name,
                    'last_name': profile.user.last_name,
                    'username': profile.user.username,
                    'email': profile.user.email,
                    'profile_picture': None,
                    'distance': "Too far"
                    # Add other profile data fields here
                })

        return Response(profile_data, status=status.HTTP_200_OK)

   


class CustomUserUpdateAPIView(APIView):
    def patch(self, request, *args, **kwargs):
        user_id = kwargs.get('pk')  # assuming you have a URL parameter for the user's ID
        try:
            user = CustomUser.objects.get(id=user_id)
        except CustomUser.DoesNotExist:
            return Response({"message": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = CustomUserSerializer(user, data=request.data, partial=True)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)





# _______________________________________________________________________________________________________________**************************************___________________________________________________________________________

stripe.api_key = settings.STRIPE_SECRET_KEY



class StripePaymentView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            # Retrieve the amount, currency, and token from the request data
            amount = request.data.get('amount', 1000)  # Amount in cents (e.g., $10.00)
            currency = request.data.get('currency', 'usd')
            token = request.data.get('token')

            # Define the return URL for the full-page redirect
            return_url = "https://example.com/checkout/success"  # Replace with your actual success URL

            # Create a Stripe Payment Intent with the provided data and return_url
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency=currency,
                payment_method=token,
                confirm=True,
                return_url=return_url  # Specify the return URL here
            )

            # Check the status of the payment and handle it as needed
            if intent.status == 'succeeded':
                # Payment was successful
                return Response({'message': 'Payment succeeded'}, status=status.HTTP_200_OK)
            elif intent.status == 'requires_action':
                # Payment requires additional actions (e.g., 3D Secure authentication)
                return Response({'client_secret': intent.client_secret, 'requires_action': True},
                                status=status.HTTP_200_OK)
            else:
                # Payment failed or has another status
                return Response({'error': 'Payment failed or has an unknown status'},
                                status=status.HTTP_400_BAD_REQUEST)

        except stripe.error.StripeError as e:
            # Handle Stripe-specific errors
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            # Handle other unexpected errors
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




class UserLikeRequestListView(generics.ListAPIView):
    serializer_class = UserLikeSerializer

    def get_queryset(self):
        # Get the liked_user_id from the query string
        liked_user_id = self.request.query_params.get('liked_user_id')

        # Check if liked_user_id is present and valid
        if liked_user_id:
            follow_request_users = UserLike.objects.filter(liked_user_id=liked_user_id, approved=False, display=True)
            return follow_request_users
        else:
            # Handle the case where liked_user_id is not provided in the query string
            return UserLike.objects.none()

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        response_data = []

        for user_like in queryset:
            user_id = user_like.user.id
            user_data = CustomUser.objects.get(id=user_id)
            user_profile_picture = ProfilePicture.objects.get(user=user_data)

            user_data_serializer = CustomUserSerializer(user_data)  # Assuming you have a serializer for your user model
            profile_picture_image = user_profile_picture.image.url if user_profile_picture.image else None

            user_like_serializer = UserLikeSerializer(user_like)  # Assuming you have a serializer for your UserLike model

            response_item = {
                'user_like': {
                    'profile_picture_image': profile_picture_image,
                    **user_data_serializer.data,  # Include fields from user_data_serializer.data
                    **user_like_serializer.data,  # Include fields from user_like_serializer.data
                },
            }

            response_data.append(response_item)

        return Response(response_data)



class CustomUserSearchByCustomIDView(APIView):

    def get(self, request, *args, **kwargs):
        custom_id = request.query_params.get('custom_id')
        current_user_id = request.query_params.get('user_id')
        if not current_user_id:
            return Response({'detail': 'user_id query parameter is required.'}, status=status.HTTP_400_BAD_REQUEST)

        if not custom_id:
            return Response({'detail': 'custom_id query parameter is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            current_user = CustomUser.objects.get(id=current_user_id)
            print('current_user.gender', current_user.gender)
            custom_user = CustomUser.objects.get(custom_id=custom_id)
            print('ustom_user.gender', custom_user.gender)
        except CustomUser.DoesNotExist:
            return Response({'detail': 'User not found.'}, status=status.HTTP_404_NOT_FOUND)

        if current_user.gender == custom_user.gender:
            raise PermissionDenied("You are not authorized to view this user's data.")
        # Search for users by custom ID
        custom_user_queryset = CustomUser.objects.filter(custom_id=custom_id)

        # Assuming you have only one ProfilePicture for the current user
        try:
            userId = CustomUser.objects.get(custom_id=custom_id)
            profile_picture = ProfilePicture.objects.get(user=userId.id)
            profile_picture_url = profile_picture.image.url
        except ProfilePicture.DoesNotExist:
            profile_picture_url = None

        try:
            profile = Profile.objects.get(user=userId.id)
        except profile.DoesNotExist:
            profile = None

        current_user_location = CustomUser.objects.get(id=current_user_id)
        requesting_user_latitude = current_user_location.latitude
        requesting_user_longitude = current_user_location.longitude
        print(requesting_user_latitude, requesting_user_longitude)
        user_data = []
        for user in custom_user_queryset:

            user_latitude = user.latitude
            user_longitude = user.longitude

            # Calculate the distance using geopy.distance
            distance = round(geodesic((requesting_user_latitude, requesting_user_longitude),
                                      (user_latitude, user_longitude)).kilometers)
            if distance is None:
                distance = "Too far"
            today = date.today()
            date_of_birth = user.date_of_birth
            if date_of_birth:
                age = today.year - date_of_birth.year - (
                        (today.month, today.day) < (date_of_birth.month, date_of_birth.day))
            else:
                age = None

            user_data.append({
                 'user_id': user.id,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'username': user.username,
                'email': user.email,
                'profile_picture': profile_picture.image.url if profile_picture.approved else None,
                'age': age,
                'distance': distance,
                'profile_for': user.profile_for,
                'date_of_birth': user.date_of_birth,
                'religion': user.religion,
                'community': user.community,
                'living_in': user.living_in,
                'mobile_number': user.mobile_number,
                'gender': user.gender,
                'latitude': user.latitude,
                'longitude': user.longitude,
                'about_me':profile.about_me,
                'marital_status': profile.marital_status,
                'height': profile.height,
                'weight': profile.weight,
                'education': profile.education,
                'occupation': profile.occupation,
                'family_status': profile.family_status,
                'income': profile.income,
                'alcoholic': profile.alcoholic,
                'smoker': profile.smoker,
                'hobbies': profile.hobbies,
                'skin_tone': profile.skin_tone,
                'time_of_bith':profile.time_of_birth
            })
        # Serialize the user data and return it as a Response
        serialized_data = CustomUserSerializer(user_data, many=True)
        return Response(user_data, status=status.HTTP_200_OK)

class CreateSubscriptionList(APIView):
    def get(self, request):
        subscriptions = CreateSubscription.objects.all()
        serializer = CreateSubscriptionSerializer(subscriptions, many=True)
        return Response(serializer.data)

class ContactUsList(APIView):
    def get(self, request):
        contacts = ContactUs.objects.all()
        serializer = ContactUsSerializer(contacts, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = ContactUsSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ContactUsDetail(APIView):
    def get_object(self, pk):
        try:
            return ContactUs.objects.get(pk=pk)
        except ContactUs.DoesNotExist:
            return None

    def get(self, request, pk):
        contact = self.get_object(pk)
        if contact is not None:
            serializer = ContactUsSerializer(contact)
            return Response(serializer.data)
        return Response(status=status.HTTP_404_NOT_FOUND)

    def put(self, request, pk):
        contact = self.get_object(pk)
        if contact is not None:
            serializer = ContactUsSerializer(contact, data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        contact = self.get_object(pk)
        if contact is not None:
            contact.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response(status=status.HTTP_404_NOT_FOUND)


class ContactDetailsView(APIView):
    def get(self, request):
        contact_details = ContactDetails.objects.first()
        if contact_details is not None:
            serializer = ContactDetailsSerializer(contact_details)
            return Response(serializer.data)
        return Response(status=404)

class SuccessStoryList(APIView):
    def get(self, request):
        success_stories = SuccessStory.objects.all()
        serializer = SuccessStorySerializer(success_stories, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = SuccessStorySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SuccessStoryDetail(APIView):
    def get_object(self, pk):
        try:
            return SuccessStory.objects.get(pk=pk)
        except SuccessStory.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        success_story = self.get_object(pk)
        serializer = SuccessStorySerializer(success_story)
        return Response(serializer.data)

    def patch(self, request, pk):
        success_story = self.get_object(pk)
        serializer = SuccessStorySerializer(success_story, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        success_story = self.get_object(pk)
        success_story.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)



class UserImagesAPI(APIView):
    def get(self, request, user_id, format=None):
        try:
            images = UploadedImages.objects.filter(user=user_id)
            serializer = UploadedImagesSerializer(images, many=True)
            return Response(serializer.data)
        except UploadedImages.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class DeleteUserImageView(APIView):
    def delete(self, request, user_id, image_id, format=None):
        try:
            # Ensure the image belongs to the specified user
            image = UploadedImages.objects.get(id=image_id, user_id=user_id)
            image.delete()
            return Response({'message': 'Image deleted successfully'}, status=status.HTTP_204_NO_CONTENT)
        except UploadedImages.DoesNotExist:
            return Response({'error': 'Image not found'}, status=status.HTTP_404_NOT_FOUND)



class FamilyNameList(APIView):
    def get(self, request, community_id):
        family_names = FamilyName.objects.filter(community__id=community_id)
        serializer = FamilyNameSerializer(family_names, many=True)
        return Response(serializer.data)




class DocumentListCreateView(APIView):
    def get(self, request):
        documents = Document.objects.all()
        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data)

    def post(self, request):
        user_id = request.data['user']
        try:
            data = Document.objects.get(id=user_id)
            return Response({"error":"User Documents Already exist"}, status=status.HTTP_400_BAD_REQUEST)
        except:

            serializer = DocumentSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DocumentDetailView(APIView):
    def get_object(self, pk):
        try:
            return Document.objects.get(pk=pk)
        except Document.DoesNotExist:
            raise status.HTTP_404_NOT_FOUND

    def get(self, request, pk):
        document = self.get_object(pk)
        serializer = DocumentSerializer(document)
        return Response(serializer.data)

    def put(self, request, pk):
        document = self.get_object(pk)
        serializer = DocumentSerializer(document, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        document = self.get_object(pk)
        document.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)




class PasswordChangeAPIView(APIView):
    def post(self, request):
        serializer = PasswordChangeSerializer(data=request.data)

        if serializer.is_valid():
            username_or_email = serializer.validated_data['username_or_email']
            new_password = serializer.validated_data['new_password']

            # Retrieve the user by username or email
            User = CustomUser
            try:
                user = User.objects.get(Q(username=username_or_email) | Q(email=username_or_email))
            except User.DoesNotExist:
                return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

            # Change the user's password
            user.set_password(new_password)
            user.save()

            return Response({"message": "Password changed successfully", "user_id": user.id}, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



class UserLikeListView(APIView):
    def get(self, request):
        # Get the current_user_id from query params
        current_user_id = request.query_params.get('current_user_id')

        if current_user_id is None:
            return Response({"error": "current_user_id is required in query params"}, status=status.HTTP_400_BAD_REQUEST)

        # Check if the user has an active subscription
        user_has_active_subscription, active_subscription = self.check_user_has_active_subscription(current_user_id)

        # Query the UserLike model to filter liked users' profiles for the current user
        liked_users = UserLike.objects.filter(user=current_user_id)

        # Create a list of dictionaries with liked user data, including uploaded images
        liked_users_data = []
        for like in liked_users:
            liked_user = like.liked_user
            profile_picture = ProfilePicture.objects.get(user=liked_user)
            uploaded_images = UploadedImages.objects.filter(user=liked_user)

            liked_user_data = {
                "custom_id": liked_user.custom_id,
                "username": liked_user.username,
                "email": liked_user.email,
                "first_name": liked_user.first_name,
                "last_name": liked_user.last_name,
                "profile_for": liked_user.profile_for,
                "image_url": profile_picture.image.url if profile_picture.approved else None,
                "uploaded_images": [image.image.url for image in uploaded_images],
                # Add other fields you need from the CustomUser model
            }
            liked_users_data.append(liked_user_data)

        # Include subscription name in the response
        response_data = {
            "subscription_name": active_subscription.subscription_type.subscription_name if user_has_active_subscription else None,
            "liked_users": liked_users_data,
        }

        return Response(response_data, status=status.HTTP_200_OK)

    def check_user_has_active_subscription(self, user_id):
        # Get the current user
        user = get_object_or_404(CustomUser, id=user_id)

        # Check if the user has an active subscription
        active_subscription = Subscription.objects.filter(user=user, end_date__gte=timezone.now()).first()

        return active_subscription is not None, active_subscription
