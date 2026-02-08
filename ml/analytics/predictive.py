"""
Predictive Activity Patterns
Analyzes historical detection data to identify recurring patterns and predict
future activity levels.

Features:
- Hourly/daily activity pattern extraction
- Anomaly vs normal activity classification
- Trend detection (increasing/decreasing/stable)
- Seasonal pattern identification
- Activity forecast for next time period
"""
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _moving_average(data: list[float], window: int = 3) -> list[float]:
    """Compute moving average with given window size."""
    if len(data) < window:
        return data
    result = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        result.append(sum(data[start:end]) / (end - start))
    return result


def _linear_trend(values: list[float]) -> tuple[float, float, str]:
    """
    Compute linear regression trend.
    Returns (slope, r_squared, trend_label).
    """
    n = len(values)
    if n < 2:
        return 0.0, 0.0, "stable"

    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n

    numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0, 0.0, "stable"

    slope = numerator / denominator

    # R-squared
    y_pred = [y_mean + slope * (x[i] - x_mean) for i in range(n)]
    ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Classify trend
    normalized_slope = slope / (y_mean + 1e-8)
    if normalized_slope > 0.05:
        trend = "increasing"
    elif normalized_slope < -0.05:
        trend = "decreasing"
    else:
        trend = "stable"

    return round(slope, 4), round(r_squared, 4), trend


def _detect_periodicity(values: list[float], max_period: int = 24) -> tuple[int, float]:
    """
    Detect periodicity in time series using autocorrelation.
    Returns (period, strength).
    """
    n = len(values)
    if n < max_period * 2:
        return 0, 0.0

    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var == 0:
        return 0, 0.0

    best_period = 0
    best_corr = 0.0

    for lag in range(2, min(max_period + 1, n // 2)):
        corr = sum(
            (values[i] - mean) * (values[i + lag] - mean)
            for i in range(n - lag)
        ) / ((n - lag) * var)

        if corr > best_corr:
            best_corr = corr
            best_period = lag

    return best_period, round(best_corr, 3)


class ActivityPatternAnalyzer:
    """
    Analyzes and stores historical activity patterns for prediction.
    """

    def __init__(self):
        # Historical data: keyed by camera/video source
        # Each entry: list of (timestamp, detection_count, event_count)
        self.history: dict[str, list[dict]] = defaultdict(list)

    def add_video_data(
        self,
        source_id: str,
        video_id: str,
        timeseries_data: dict,
        events: list[dict],
        video_start_time: Optional[datetime] = None,
    ):
        """
        Add processed video data to the historical record.

        Args:
            source_id: Camera/source identifier (for grouping)
            video_id: Video identifier
            timeseries_data: Output from timeseries.generate_timeseries()
            events: List of event dicts
            video_start_time: Real-world start time (if known)
        """
        timestamps = timeseries_data.get("timestamps", [])
        total_dets = timeseries_data.get("total_detections", [])

        for i, ts in enumerate(timestamps):
            real_time = None
            if video_start_time:
                real_time = video_start_time + timedelta(seconds=ts)

            self.history[source_id].append({
                "video_id": video_id,
                "relative_sec": ts,
                "real_time": real_time.isoformat() if real_time else None,
                "hour_of_day": real_time.hour if real_time else None,
                "day_of_week": real_time.weekday() if real_time else None,
                "detection_count": total_dets[i] if i < len(total_dets) else 0,
            })

        # Store events
        event_counts = Counter(e.get("event_type", "unknown") for e in events)
        self.history[f"{source_id}__events"].append({
            "video_id": video_id,
            "event_counts": dict(event_counts),
            "total_events": len(events),
            "real_time": video_start_time.isoformat() if video_start_time else None,
        })

        logger.info(
            f"Pattern analyzer: Added {len(timestamps)} data points "
            f"from source '{source_id}'"
        )

    def analyze_patterns(self, source_id: str) -> dict:
        """
        Analyze activity patterns for a source.

        Returns:
            {
                "hourly_pattern": {0-23: avg_activity},
                "trend": {"slope": float, "direction": str, "r_squared": float},
                "periodicity": {"period": int, "strength": float},
                "baseline": {"mean": float, "std": float, "min": float, "max": float},
                "anomaly_thresholds": {"low": float, "high": float},
                "forecast": [predicted_values for next N periods],
                "peak_hours": [hour_ints],
                "quiet_hours": [hour_ints],
            }
        """
        data = self.history.get(source_id, [])
        if not data:
            return self._empty_pattern()

        # Extract detection counts
        counts = [d["detection_count"] for d in data]

        # Baseline statistics
        mean_val = sum(counts) / len(counts)
        std_val = math.sqrt(sum((c - mean_val) ** 2 for c in counts) / len(counts)) if len(counts) > 1 else 0

        # Trend analysis
        slope, r_squared, trend_dir = _linear_trend(counts)

        # Periodicity detection
        period, period_strength = _detect_periodicity(counts)

        # Hourly patterns (if real timestamps available)
        hourly_pattern = {}
        hourly_data = defaultdict(list)
        for d in data:
            hour = d.get("hour_of_day")
            if hour is not None:
                hourly_data[hour].append(d["detection_count"])

        for hour in range(24):
            vals = hourly_data.get(hour, [])
            hourly_pattern[hour] = round(sum(vals) / len(vals), 1) if vals else 0

        # Peak and quiet hours
        if hourly_pattern and any(v > 0 for v in hourly_pattern.values()):
            avg_hourly = sum(hourly_pattern.values()) / 24
            peak_hours = [h for h, v in hourly_pattern.items() if v > avg_hourly * 1.5]
            quiet_hours = [h for h, v in hourly_pattern.items() if 0 < v < avg_hourly * 0.5]
        else:
            peak_hours = []
            quiet_hours = []

        # Anomaly thresholds (2 standard deviations)
        anomaly_low = max(0, mean_val - 2 * std_val)
        anomaly_high = mean_val + 2 * std_val

        # Simple forecast (moving average extrapolation)
        smoothed = _moving_average(counts, window=5)
        forecast = []
        if len(smoothed) >= 3:
            last_3 = smoothed[-3:]
            avg_recent = sum(last_3) / 3
            # Project forward with trend
            for i in range(1, 11):
                predicted = avg_recent + slope * i
                forecast.append(round(max(0, predicted), 1))

        result = {
            "hourly_pattern": hourly_pattern,
            "trend": {
                "slope": slope,
                "direction": trend_dir,
                "r_squared": r_squared,
            },
            "periodicity": {
                "period": period,
                "strength": period_strength,
                "description": (
                    f"Repeating pattern every {period} intervals (strength: {period_strength:.0%})"
                    if period_strength > 0.3 else "No significant periodicity detected"
                ),
            },
            "baseline": {
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "min": min(counts),
                "max": max(counts),
                "total_observations": len(counts),
            },
            "anomaly_thresholds": {
                "low": round(anomaly_low, 2),
                "high": round(anomaly_high, 2),
            },
            "forecast": forecast,
            "peak_hours": sorted(peak_hours),
            "quiet_hours": sorted(quiet_hours),
        }

        logger.info(
            f"Pattern analysis for '{source_id}': trend={trend_dir}, "
            f"period={period} (strength={period_strength}), "
            f"baseline={mean_val:.1f} +/- {std_val:.1f}"
        )

        return result

    def classify_activity_level(
        self,
        source_id: str,
        current_count: float,
    ) -> dict:
        """
        Classify if current activity is normal, low, or high relative to history.

        Returns:
            {
                "level": "normal" | "low" | "high" | "anomalous",
                "z_score": float,
                "percentile": float,
                "description": str,
            }
        """
        data = self.history.get(source_id, [])
        if not data:
            return {
                "level": "unknown",
                "z_score": 0,
                "percentile": 50,
                "description": "Insufficient historical data",
            }

        counts = sorted(d["detection_count"] for d in data)
        mean_val = sum(counts) / len(counts)
        std_val = math.sqrt(sum((c - mean_val) ** 2 for c in counts) / len(counts))

        z_score = (current_count - mean_val) / std_val if std_val > 0 else 0

        # Percentile
        below = sum(1 for c in counts if c <= current_count)
        percentile = (below / len(counts)) * 100

        if abs(z_score) > 3:
            level = "anomalous"
            desc = f"Highly unusual activity ({z_score:+.1f} sigma from normal)"
        elif z_score > 2:
            level = "high"
            desc = f"Above-average activity (top {100 - percentile:.0f}%)"
        elif z_score < -2:
            level = "low"
            desc = f"Below-average activity (bottom {percentile:.0f}%)"
        else:
            level = "normal"
            desc = f"Normal activity level ({percentile:.0f}th percentile)"

        return {
            "level": level,
            "z_score": round(z_score, 2),
            "percentile": round(percentile, 1),
            "description": desc,
        }

    def _empty_pattern(self) -> dict:
        return {
            "hourly_pattern": {h: 0 for h in range(24)},
            "trend": {"slope": 0, "direction": "unknown", "r_squared": 0},
            "periodicity": {"period": 0, "strength": 0, "description": "No data"},
            "baseline": {"mean": 0, "std": 0, "min": 0, "max": 0, "total_observations": 0},
            "anomaly_thresholds": {"low": 0, "high": 0},
            "forecast": [],
            "peak_hours": [],
            "quiet_hours": [],
        }


# Singleton analyzer
_analyzer: Optional[ActivityPatternAnalyzer] = None


def get_pattern_analyzer() -> ActivityPatternAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ActivityPatternAnalyzer()
    return _analyzer


def analyze_video_patterns(
    video_id: str,
    timeseries_data: dict,
    events: list[dict],
    source_id: str = "default",
    video_start_time: Optional[datetime] = None,
) -> dict:
    """
    Convenience function: add video data and return pattern analysis.
    """
    analyzer = get_pattern_analyzer()

    analyzer.add_video_data(
        source_id=source_id,
        video_id=video_id,
        timeseries_data=timeseries_data,
        events=events,
        video_start_time=video_start_time,
    )

    return analyzer.analyze_patterns(source_id)
