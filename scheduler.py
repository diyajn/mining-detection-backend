"""
Daily Scheduler - Runs monitoring at 3:00 AM
Keep this running in background
"""

import schedule
import time
from automated_monitor import AutomatedMonitor

def daily_monitoring_job():
    """Run at 3:00 AM daily"""
    print(f"\n⏰ Daily scan triggered at 3:00 AM")
    monitor = AutomatedMonitor()
    monitor.run_daily_monitoring()



# Original (runs at 3:00 AM):
schedule.every().day.at("16:20").do(daily_monitoring_job)

# Change to run in 2 minutes from now:(for testing purpose)
# import datetime
# now = datetime.datetime.now()
# test_time = (now + datetime.timedelta(minutes=1)).strftime("%H:%M")
# schedule.every().day.at(test_time).do(daily_monitoring_job)
# print(f"🧪 TEST MODE: Will run at {test_time}")




print("="*70)
print("🕐 DAILY SCHEDULER STARTED")
print("="*70)
print("\n✅ Will run daily monitoring at 3:00 AM")
print("⏰ Current time:", time.strftime("%H:%M:%S"))
print("\n💡 Keep this window open (minimize it)")
print("   Press Ctrl+C to stop\n")
print("="*70 + "\n")

while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute