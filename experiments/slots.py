import datetime as dt, sys
# Free submit slots from now to 28 Oct (29 and 30 Oct stay free for the final notebook output).
now = dt.datetime.now()
slots = [dt.datetime(2026, 10, d, h, 1) for d in range(13, 29) for h in (7, 12, 18)]
slots = [s for s in slots if s > now + dt.timedelta(minutes=5)]
if sys.argv[1:] == ["count"]: print(len(slots))
else: print("\n".join(s.strftime("%Y-%m-%d %H:%M") for s in slots))
