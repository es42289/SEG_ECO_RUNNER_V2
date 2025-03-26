from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.utils.dateparse import parse_date
from snowflake.snowpark import Session
from .snowflake_config import conn

def index(request):
    return render(request, 'graph/index.html', {'position': position})

position = {'x': 0, 'y': 1000, 'Di': 0.15, 'b': 0.5}

def update_position(request):
    global position
    position['x'] = float(request.GET.get('x', position['x']))
    position['y'] = float(request.GET.get('y', position['y']))
    position['Di'] = float(request.GET.get('Di', position['Di']))
    position['b'] = float(request.GET.get('b', position['b']))
    return JsonResponse(position)

def get_production_data(api_uwi):
    """Get production data for a well"""
    try:
        session = Session.builder.configs(conn).create()
        query = f"""
        SELECT API_UWI, ProducingMonth, LIQUIDSPROD_BBL, GASPROD_MCF
        FROM wells.minerals.raw_prod_data
        WHERE API_UWI = '{api_uwi}'
        ORDER BY ProducingMonth
        """
        result = session.sql(query).to_pandas()
        return result
    except Exception as e:
        return pd.DataFrame()  # fail gracefully
    
@csrf_exempt
def get_production_json(request):
    api_uwi = request.GET.get('api_uwi', '42-041-30689')
    df = get_production_data(api_uwi)

    if df.empty:
        return JsonResponse({'error': 'No data found'}, status=404)

    data = [
        {'x': pd.to_datetime(row['ProducingMonth']).strftime('%Y-%m-%d'), 'y': row['LIQUIDSPROD_BBL']}
        for _, row in df.iterrows()
    ]

    return JsonResponse({'data': data})

def get_fast_edit_wells():
    """Get all wells with FAST_EDIT=1"""
    # try:
    session = Session.builder.configs(conn).create()
    
    # Join ECON_INPUT with vw_well to get well names and trajectory
    df_snow = session.sql("""
    SELECT e.*, w.WELLNAME, w.TRAJECTORY
    FROM wells.minerals.ECON_INPUT e
    JOIN wells.minerals.vw_well_input w
    ON e.API_UWI = w.API_UWI
    WHERE e.FAST_EDIT = 1
    ORDER BY w.WELLNAME
    """)
    rows = df_snow.collect()  # returns list of Row objects
    records = [r.as_dict() for r in rows]
    return pd.DataFrame.from_records(records)

@csrf_exempt
def get_fast_edit_wells_json(request):
    try:
        df = get_fast_edit_wells()
        wells = [
            {
                "api": row["API_UWI"],
                "name": row["WELLNAME"],
                "trajectory": row["TRAJECTORY"]
            }
            for _, row in df.iterrows()
        ]
        return JsonResponse({"wells": wells})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

