from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

from .services.ml_engine import predict_event_impact


DEFAULT_FORM = {
    "event_type": "unplanned",
    "event_cause": "vehicle_breakdown",
    "priority": "High",
    "latitude": "12.9716",
    "longitude": "77.5946",
    "end_latitude": "",
    "end_longitude": "",
    "corridor": "Non-corridor",
    "veh_type": "",
    "police_station": "",
    "timestamp": "",
    "requires_road_closure": "no",
}


def dashboard_view(request):

    result = None
    error = None

    form_data = DEFAULT_FORM.copy()

    if request.method == "POST":

        form_data.update(
            request.POST.dict()
        )

        try:

            result = predict_event_impact(
                form_data
            )

        except Exception as e:

            error = str(e)

    return render(
        request,
        "dashboard/index.html",
        {
            "form": form_data,
            "result": result,
            "error": error,
        }
    )
