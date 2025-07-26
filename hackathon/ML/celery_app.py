from celery import Celery
import os
from celery.schedules import crontab
from celery.schedules import crontab

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

cel = Celery(
    'bharati_outreach',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['tasks', 'reply_listener']  # Include both task modules
)

# Define the periodic task schedule (Celery Beat)
cel.conf.beat_schedule = {
    'check-for-replies-every-5-minutes': {
        'task': 'tasks.check_for_replies', # The full path to the task
        'schedule': crontab(minute='*/5'),  # Run every 5 minutes
    },
}
cel.conf.timezone = 'UTC'

# Configuration
cel.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)