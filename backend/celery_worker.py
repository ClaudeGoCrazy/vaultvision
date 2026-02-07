"""Celery worker entry point.

Run with:
    celery -A celery_worker.celery_app worker --loglevel=info
"""
from app.services.tasks import celery_app
