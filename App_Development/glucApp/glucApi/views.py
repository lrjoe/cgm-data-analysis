# from django.shortcuts import render
# from django.http import JsonResponse
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# from rest_framework import status
# from glucApi.models import Glucose
# from glucApi.serializer import GlucoseSerializer
# # Create your views here.


from typing import Any
from rest_framework.views import APIView
from glucApi.models import Glucose
from glucApi.serializer import GlucoseSerializer
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
import boto3
#from boto.s3.connection import S3Connection
from glucApp import settings 
import pandas as pd
import numpy as np
import io
import os
from datetime import date, timedelta
from pyspark.ml import PipelineModel, Pipeline
import eli5
from pyspark.ml.util import DefaultParamsReader
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast, TYPE_CHECKING
from pyspark.sql import SparkSession
from pyspark.sql.functions import max, lit


class PatientForm(APIView):
    def __init__(self):
        self.filename = 'index.html'

    def get(self, request):

        # conn = S3Connection(settings.AWS_ACCESS_KEY_ID,settings.AWS_SECRET_ACCESS_KEY)
        # bucket = conn.get_bucket(settings.MEDIA_BUCKET)
       
        session = boto3.Session(aws_access_key_id = settings.ACCESS_ID,
                         aws_secret_access_key = settings.SECRET_ACCESS_KEY)

        s3 = session.resource('s3', endpoint_url = settings.ENDPOINT_URL)

        my_bucket = s3.Bucket(settings.STORAGE_BUCKET_NAME)

        return render(request,template_name=self.filename)
    
class TestingStuffWithkarina(APIView):
    def get(self, request):
        pipelineModel=PipelineModel.load("/home/katie/Documents/DSE/DSE260A/Capstone/glucose-data-analysis/App_Development/glucApp/XGBClassification")
        print('did it get here????')

class GetPatientInfo(APIView):
    def get(self, request):
        patientId = int(request.GET['patientId'])
        patientPassword = request.GET['patientPassword']

        print('patientId', patientId)
        print('patientPassword', patientPassword)

        s3_client = boto3.client('s3', aws_access_key_id = settings.ACCESS_ID,
                        aws_secret_access_key = settings.SECRET_ACCESS_KEY,
                        endpoint_url = settings.ENDPOINT_URL)
       
        spark = SparkSession.builder.getOrCreate()

        df = pd.DataFrame()
        fileNames = s3_client.list_objects_v2(Bucket=settings.STORAGE_BUCKET_NAME)
        fileNames = fileNames['Contents']
        fileNames = [x for x in fileNames if ('parquet' in x['Key'])\
                      and ('five_patients_full' in x['Key'])\
                      and ('.crc' not in x['Key'])]
        for fileObj in fileNames:
            key = fileObj['Key']
            obj = s3_client.get_object(Bucket=settings.STORAGE_BUCKET_NAME,\
                                        Key=fileObj['Key'])
            
            parquet_df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
            df = pd.concat([df, parquet_df[parquet_df.NumId == patientId]])

        patientRow =spark.createDataFrame(df) 
        
        pipelineModel=PipelineModel.load("/home/katie/Documents/DSE/DSE260A/Capstone/glucose-data-analysis/glucApp/cephfsModel/XGBClassification7")
        print('gobble gobble')
        preds=pipelineModel.transform(patientRow)
        # preds.select('NumId', 'Chunk', 'rawPrediction', 'probability', 'prediction', 'target').show()

        max_val = preds.select(max(preds.Chunk)).collect()
        final_val = preds.filter(preds.Chunk == lit(max_val[0][0]))

        probability = final_val.select('probability').collect()[0][0]

        prediction = int(final_val.select('prediction').collect()[0][0])

        sum_test_pandas=df[df.Chunk == max_val[0][0]]

        sum_test_pandas=sum_test_pandas[['ShortTermVariance', 'LongTermVariance', 'VarianceRatio', 'SampleEntropy', 
                                        'PermutationEntropy', 'Mean', 'StdDev', 'Median', 'Min', 'Max', 'AvgFirstDiff', 
                                        'AvgSecDiff', 'StdFirstDiff', 'StdSecDiff', 'CountAbove', 'CountBelow', 'TotalOutOfRange',
                                        'Sex_Female', 'Sex_Male', 'Treatment_yes_both', 'Treatment_yes_long_acting','Treatment_no',
                                        'Treatment_yes_fast_acting','AgeGroup_50','AgeGroup_60',
                                        'AgeGroup_70','AgeGroup_40','AgeGroup_30','AgeGroup_80','AgeGroup_90','AgeGroup_10']]
        
        target_arr=sum_test_pandas.values[0]

        xgb_model=pipelineModel.stages[-1].bestModel.get_booster()

        feature_mapping_eli5={'x0': 'scaled_ShortTermVariance', 'x1': 'scaled_LongTermVariance', 'x2':'scaled_VarianceRatio',
            'x3': 'scaled_SampleEntropy', 'x4': 'scaled_PermutationEntropy', 'x5': 'scaled_Mean',
            'x6': 'scaled_StdDev', 'x7': 'scaled_Median', 'x8': 'scaled_Min',
            'x9': 'scaled_Max', 'x10': 'scaled_AvgFirstDiff', 'x11': 'scaled_AvgSecDiff',
            'x12': 'scaled_StdFirstDiff', 'x13': 'scaled_StdSecDiff', 'x14': 'scaled_CountAbove',
            'x15': 'scaled_CountBelow', 'x16': 'scaled_TotalOutOfRange', 'x17': 'Sex_Female',
            'x18': 'Sex_Male', 'x19': 'Treatment_yes_both', 'x20': 'Treatment_yes_long_acting',
            'x21': 'Treatment_no', 'x22': 'Treatment_yes_fast_acting', 'x23': 'AgeGroup_50',
            'x24':'AgeGroup_60', 'x25': 'AgeGroup_70', 'x26': 'AgeGroup_40',
            'x27': 'AgeGroup_30', 'x28': 'AgeGroup_80','x29': 'AgeGroup_90',
            'x30': 'AgeGroup_10', '<BIAS>': 'Default Value'}

        prediction_vals=eli5.explain_prediction_df(xgb_model, target_arr)

        prediction_vals['feature_name']=prediction_vals['feature'].map(feature_mapping_eli5)

        top_five_feats=prediction_vals[(prediction_vals['target']==int(prediction))\
                                       & (prediction_vals['feature'] != '<BIAS>')]\
            .sort_values(by='weight', ascending=False)\
            .reset_index(drop=True)
        
        top_five_feats=top_five_feats.iloc[0:5, :].reset_index(drop=True)

        top_five_feats['feature_name']=top_five_feats['feature'].map(feature_mapping_eli5)

        feats_json = top_five_feats.feature_name.to_json()        

        return JsonResponse({ 'success': True, 
                              'probability': int(probability[prediction] * 100),
                              'prediction': prediction,
                              'feats' : feats_json
                              })

class PatientList(APIView):
    def get(self, request):
        glucs = Glucose.objects.all() # complex data
        glucSerialized = GlucoseSerializer(glucs, many=True)
        
        return Response(glucSerialized.data)
    
    def post(self, request):
        return Response({ "hello": "friend"})
    

class PatientCreate(APIView):
    def post(self, request):
        #create gluocose values (maybe when we add new user)

        serializer = GlucoseSerializer(data=request.data)

        if serializer.is_valid():
            serializer.save() # save to db
            return Response(serializer.data)
        else:
            return Response(serializer.errors)


class PatientGlucose(APIView):
    def get_gluc_from_pk(self, patientId):
        try:
            return Glucose.objects.get(PatientId=patientId)
        except:
            return Response({
                "error": "Patient is not found"
            }, status=status.HTTP_404_NOT_FOUND)

    def get(self, request, pk):
        glucVals = self.get_gluc_from_pk(pk);
        serializer = GlucoseSerializer(glucVals)
        return Response(serializer.data)


    def post(self, request, pk):
        #create gluocose values (maybe when we add new user)
        
        glucVals = self.get_gluc_from_pk(pk)

        serializer = GlucoseSerializer(glucVals)

        if serializer.is_valid():
            serializer.save() # save to db
            return Response(serializer.data)
        else:
            return Response(serializer.errors)
        
    def put(self, request, pk):
        glucVals = self.get_gluc_from_pk(pk)
        serializer = GlucoseSerializer(glucVals, data=request.data)
        if serializer.is_valid:
            serializer.save()
            return Response(serializer.data)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request, pk):
        glucVals = self.get_gluc_from_pk(pk)
        #glucVals.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


