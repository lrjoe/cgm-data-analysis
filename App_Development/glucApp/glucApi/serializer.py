# from rest_framework import serializers
# from glucApi.models import Glucose

# # needed: pip install djangorestframework

# class GlucoseSerializer(serializers.Serializer):
#     id = serializers.IntegerField(read_only=True)
#     PatientId = serializers.CharField(max_length=100);
#     GlucoseDisplayTime = serializers.CharField()
#     Value = serializers.IntegerField()

#     def create(self, data):
#         return Glucose.objects.create(**data)
    
#     def update(self, instance, data):
#         instance.PatientId = data.get('PatientId', instance.PatientId)
#         instance.GlucoseDisplayTime = data.get('GlucoseDisplayTime', instance.GlucoseDisplayTime)
#         instance.Value = data.get('Value', instance.Value)

#         instance.save()

#         return instance;

from rest_framework import serializers
from glucApi.models import Glucose
from django.forms import ValidationError

class GlucoseSerializer(serializers.ModelSerializer):
    description = serializers.SerializerMethodField()

    class Meta:
        model = Glucose
        fields = '__all__'

    def validate_Value(self, value):
        if value < 0:
            raise ValidationError("Value cannot be less than 0")
        return value
    
    def validate(self, data):
        if data['PatientId'] < 0:
            raise ValidationError("Patient Id cannot be less than 0")
        return data
    
    def get_description(self, data):
        return "This patient is super healthy. THeir range is " + str(data.Value.max()) + \
                " and " + str(data.Value.min())