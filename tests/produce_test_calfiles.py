import hickle
from astropy import units as u
from edges_cal import CalibrationObservation
from edges_io import io

calio = io.CalibrationObservation(
    "/data5/edges/data/CalibrationObservations/Receiver01/Receiver01_25C_2015_09_02_040_to_200MHz/",
    run_num={"receiver_reading": (1, 2, 4, 6, 8)},
    repeat_num=1,
)

calobs = CalibrationObservation.from_io(
    calio,
    f_low=50.0 * u.MHz,
    f_high=100.0 * u.MHz,
    cterms=6,
    wterms=5,
    spectrum_kwargs={
        "default": {"t_load": 300, "t_load_ns": 350, "ignore_times_percent": 7},
        "hot_load": {"ignore_times_percent": 10},
    },
    receiver_kwargs={"n_terms": 11, "model_type": "polynomial"},
)

hickle.dump(calobs, "data/test_calfile.h5")
