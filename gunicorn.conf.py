from apscheduler.schedulers.background import BackgroundScheduler
from internal.kb_helper import run_conf_refresh_tasks

# Define your Gunicorn settings
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
preload_app = True

# Create a single scheduler instance
scheduler = BackgroundScheduler()

def when_ready(server):
    """
    This hook is called by the Gunicorn master process when it's ready.
    """
    print("Server is ready. Starting the scheduler.")
    # Add your job and start the scheduler
    scheduler.add_job(run_conf_refresh_tasks, 'cron', hour=1, minute=30)
    scheduler.start()

def on_exit(server):
    """
    This hook is called by the Gunicorn master process when it's exiting.
    """
    print("Server is exiting. Shutting down the scheduler.")
    scheduler.shutdown()
