from django.db import models

# https://www.youtube.com/watch?v=mlr9BF4JomE
# Create your models here.
# Create objects that relate to data in our db

class Glucose(models.Model):
    PatientId = models.CharField(max_length=100);
    GlucoseDisplayTime = models.DateTimeField()
    Value = models.IntegerField()

    def __str__(self):
        return self.PatientId

class GlucoseFull(models.Model):
    PatientId = models.CharField(max_length=100)
    GlucoseDisplayTime = models.DateTimeField()
    Value = models.IntegerField()

    PostDate = models.DateField()
    IngestionDate = models.DateField()
    PostId = models.CharField(max_length=100)
    PostTime = models.DateTimeField()
    Stream = models.CharField(max_length=100)
    SequenceNumber = models.IntegerField()
    TransmitterNumber = models.CharField(max_length=100)
    ReceiverNumber = models.CharField(max_length=100)
    RecordedSystemTime = models.DateTimeField()
    RecordedDisplayTime = models.DateTimeField()
    RecordedDisplayTimeRaw = models.DateTimeField()
    TransmitterId = models.CharField(max_length=100)
    TransmitterTime = models.IntegerField()
    GlucoseSystemTime = models.DateTimeField()
    GlucoseDisplayTime = models.DateTimeField()
    GlucoseDisplayTimeRaw = models.DateTimeField()
    Status = models.CharField(max_length=100)
    TrendArrow = models.CharField(max_length=100)
    TrendRate = models.FloatField()
    IsBackFilled = models.BooleanField()
    InternalStatus = models.IntegerField()
    SessionStartTime = models.IntegerField()


# plain raw data format
class GlucoseRaw(models.Model):
    PatientId = models.CharField(max_length=100)
    GlucoseDisplayTime = models.DateTimeField()
    Value = models.IntegerField()

    PostDate = models.CharField(max_length=100)
    IngestionDate = models.DateField()
    PostId = models.CharField(max_length=100)
    PostTime = models.DateTimeField()
    Stream = models.CharField(max_length=100)
    SequenceNumber = models.IntegerField()
    TransmitterNumber = models.CharField(max_length=100)
    ReceiverNumber = models.CharField(max_length=100)
    RecordedSystemTime = models.CharField(max_length=100)
    RecordedDisplayTime = models.CharField(max_length=100)
    RecordedDisplayTimeRaw = models.CharField(max_length=100)
    TransmitterId = models.CharField(max_length=100)
    TransmitterTime = models.IntegerField()
    GlucoseSystemTime = models.CharField(max_length=100)
    GlucoseDisplayTime = models.CharField(max_length=100)
    GlucoseDisplayTimeRaw = models.CharField(max_length=100)
    Status = models.CharField(max_length=100)
    TrendArrow = models.CharField(max_length=100)
    TrendRate = models.FloatField()
    IsBackFilled = models.BooleanField()
    InternalStatus = models.IntegerField()
    SessionStartTime = models.IntegerField()




