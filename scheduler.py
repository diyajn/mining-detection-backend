import schedule
import time
import datetime
from automated_monitor import AutomatedMonitor

def daily_monitoring_job():
    print(f"\n⏰ Daily scan triggered at scheduled time!")
    monitor = AutomatedMonitor()
    monitor.run_daily_monitoring()

def start_scheduler():
    # Set your fixed scheduled time here in 24-hour HH:MM format
    scheduled_time = "20:20"  # Change this to your desired daily time
    
    # For testing, you can uncomment below to run 2 minutes from now:
    # now = datetime.datetime.now()
    # scheduled_time = (now + datetime.timedelta(minutes=2)).strftime("%H:%M")

    print(f"🧪 Scheduler will run daily monitoring at {scheduled_time}")

    # Schedule the daily job at this time
    schedule.every().day.at(scheduled_time).do(daily_monitoring_job)

    print("🕐 DAILY SCHEDULER STARTED - Waiting for scheduled time...")

    while True:
        schedule.run_pending()
        time.sleep(60)  # check every second for precise triggering
