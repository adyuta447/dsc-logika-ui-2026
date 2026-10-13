import datetime as dt, sys
# Free submit slots from now to 30 Oct 12:01; 30 Oct 18:01 stays free as a reserve for the final notebook output (bag v4 top_mix is already submitted).
now = dt.datetime.now()
# 14 Oct starts later so the overnight training and simulation finish first; the quota resets at 07:00, so three later slots still fit that day
hours = lambda d: (10, 15, 20) if d == 14 else (7, 12) if d == 30 else (7, 12, 18)
allslots = [dt.datetime(2026, 10, d, h, 1) for d in range(13, 31) for h in hours(d)]
slots = [s for s in allslots if s > now + dt.timedelta(minutes=5)]
if sys.argv[1:] == ["count"]: print(len(slots))
# busy: exit 1 from 15 minutes before to 40 minutes after a slot, while the scheduler may be submitting
elif sys.argv[1:] == ["busy"]: sys.exit(any(s - dt.timedelta(minutes=15) <= now <= s + dt.timedelta(minutes=40) for s in allslots))
else: print("\n".join(s.strftime("%Y-%m-%d %H:%M") for s in slots))
