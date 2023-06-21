from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path('glucApp/', include('glucApi.urls'))
    #path('react/', include('reactApp.urls'))
]
