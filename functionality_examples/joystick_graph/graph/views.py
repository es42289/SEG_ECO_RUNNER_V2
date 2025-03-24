from django.shortcuts import render
from django.http import JsonResponse

# position = {'x': 0, 'y': 0, 'm': 1}

def index(request):
    return render(request, 'graph/index.html', {'position': position})

# def update_position(request):
#     global position
#     x = float(request.GET.get('x', position['x']))
#     y = float(request.GET.get('y', position['y']))
#     m = float(request.GET.get('m', position['m']))
#     position.update({'x': x, 'y': y, 'm': m})
#     return JsonResponse(position)
position = {'x': 0, 'y': 1000, 'Di': 0.15, 'b': 0.5}

def update_position(request):
    global position
    position['x'] = float(request.GET.get('x', position['x']))
    position['y'] = float(request.GET.get('y', position['y']))
    position['Di'] = float(request.GET.get('Di', position['Di']))
    position['b'] = float(request.GET.get('b', position['b']))
    return JsonResponse(position)
