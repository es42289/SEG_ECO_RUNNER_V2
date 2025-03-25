from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('update/', views.update_position, name='update'),
    path('get_production/', views.get_production_json),  # 👈 new route
    path('api/fast_edit_wells/', views.get_fast_edit_wells_json),
]
