import numpy as np


class HurdleModelBundle(dict):
    """
    Dict-like hurdle model package with sklearn-style .predict() support.

    This keeps both things working:

    1. New hurdle architecture:
       classifier + regressor + threshold

    2. Old model-style usage:
       model.predict(X)
    """

    def predict(self, X):

        result = predict_forecast_count(
            self,
            X
        )

        return result["expected_count"]

    def predict_details(self, X):

        return predict_forecast_count(
            self,
            X
        )

    def predict_proba(self, X):

        classifier = self["classifier"]

        return classifier.predict_proba(
            X
        )


def predict_forecast_count(
    model_bundle,
    X
):
    """
    Supports:
    1. HurdleModelBundle
    2. plain dict hurdle bundle
    3. old single CatBoostRegressor
    """

    # =====================================================
    # ZERO-INFLATED / HURDLE MODEL
    # =====================================================

    if (
        isinstance(model_bundle, dict)
        and model_bundle.get("model_type") == "zero_inflated_hurdle_v1"
    ):

        classifier = model_bundle["classifier"]
        regressor = model_bundle["regressor"]

        alert_threshold = model_bundle.get(
            "alert_threshold",
            0.55
        )

        positive_count_mean = model_bundle.get(
            "positive_count_mean",
            1.0
        )

        alert_proba = classifier.predict_proba(X)[:, 1]

        alert_pred = (
            alert_proba
            >=
            alert_threshold
        ).astype(int)

        # =================================================
        # Positive count prediction
        # =================================================

        if regressor is not None:

            positive_log_pred = regressor.predict(
                X
            )

            positive_count_pred = np.expm1(
                positive_log_pred
            )

            positive_count_pred = np.maximum(
                positive_count_pred,
                0.0
            )

        else:

            positive_count_pred = np.full(
                len(X),
                positive_count_mean
            )

        # =================================================
        # Threshold-gated count prediction
        #
        # Because 93%+ rows are zero, we should not predict
        # count unless alert probability crosses threshold.
        # =================================================

        probability_strength = (
            alert_proba
            -
            alert_threshold
        ) / (
            1.0
            -
            alert_threshold
            +
            1e-9
        )

        probability_strength = np.clip(
            probability_strength,
            0.0,
            1.0
        )

        expected_count = (
            probability_strength
            *
            positive_count_pred
        )

        expected_count = np.maximum(
            expected_count,
            0.0
        )

        raw_expected_count = (
            alert_proba
            *
            positive_count_pred
        )

        raw_expected_count = np.maximum(
            raw_expected_count,
            0.0
        )

        return {
            "expected_count": expected_count,
            "raw_expected_count": raw_expected_count,
            "alert_probability": alert_proba,
            "alert_prediction": alert_pred,
            "positive_count_prediction": positive_count_pred,
            "model_type": "zero_inflated_hurdle_v1"
        }

    # =====================================================
    # OLD SINGLE REGRESSOR FALLBACK
    # =====================================================

    preds = model_bundle.predict(
        X
    )

    preds = np.maximum(
        preds,
        0.0
    )

    return {
        "expected_count": preds,
        "raw_expected_count": preds,
        "alert_probability": None,
        "alert_prediction": None,
        "positive_count_prediction": None,
        "model_type": "single_regressor"
    }


def predict_single_forecast(
    model_bundle,
    X
):

    result = predict_forecast_count(
        model_bundle,
        X
    )

    return (
        float(result["expected_count"][0]),
        result
    )