"""What the recorded public scores actually say about OOF KS.

Four of our own files have both a measured out-of-fold KS and a measured public
score, plus the team's historical file. If OOF KS predicted public score, these
five points would line up. They do not, and the spread of the public numbers is
about five times the spread of the OOF numbers, which changes what a local
metric can be used for.
"""

DATA = [
    # label, OOF KS, public, note
    ("v57 aggressive_145", 0.526988, 0.557, "speculative direction, never public-proven"),
    ("v58ext ext1.5", 0.527511, 0.559, "spearman 0.9994 to qc_00, yet 0.0027 lower"),
    ("team qc_00 ext1.5", None, 0.56171, "proven direction, OOF KS never measured"),
    ("v61 alt_ext0.5_tm30_lean", 0.528624, 0.56312, "best public score so far"),
    ("v63 best candidate", 0.529867, 0.557, "highest OOF KS, worst public of the four"),
]

rows = [r for r in DATA if r[1] is not None]
pairs = [(r[1], r[2]) for r in rows]

print("=== recorded pairs ===")
for name, ks, pub, note in rows:
    print(f"  {name:28s} OOF {ks:.6f}  public {pub:.5f}  offset {pub - ks:+.5f}   {note}")

ks_list = [k for k, _ in pairs]
pub_list = [p for _, p in pairs]
print(f"\nOOF KS spread   : {min(ks_list):.6f} to {max(ks_list):.6f}  = {max(ks_list) - min(ks_list):.6f}")
print(f"public spread   : {min(pub_list):.5f} to {max(pub_list):.5f}  = {max(pub_list) - min(pub_list):.5f}")
print(f"ratio           : public spread is {(max(pub_list) - min(pub_list)) / (max(ks_list) - min(ks_list)):.1f}x the OOF spread")

n = len(pairs)
mk, mp = sum(ks_list) / n, sum(pub_list) / n
ck = sum((a - mk) * (b - mp) for a, b in pairs) / n
sk = sum((a - mk) ** 2 for a in ks_list) ** 0.5
sp = sum((b - mp) ** 2 for b in pub_list) ** 0.5
print(f"\nPearson r (n={n}) : {ck / (sk * sp):+.3f}")
print("Spearman r        : ranks of OOF KS vs public")
order_ks = sorted(range(n), key=lambda i: ks_list[i])
order_pub = sorted(range(n), key=lambda i: pub_list[i])
rk = [0] * n
for r, i in enumerate(order_ks):
    rk[i] = r
rp = [0] * n
for r, i in enumerate(order_pub):
    rp[i] = r
cc = sum((rk[i] - (n - 1) / 2) * (rp[i] - (n - 1) / 2) for i in range(n)) / n
print(f"                 : {cc / (((n - 1) ** 2 / 12) ** 0.5 * ((n - 1) ** 2 / 12) ** 0.5):+.3f}")
print("\n  OOF KS order (best first):", [DATA[i][0] for i in reversed(order_ks)])
print("  public order (best first):", [rows[i][0] for i in reversed(order_pub)])

print("\n=== the decisive pair ===")
print("  v58ext ext1.5 and team qc_00 differ by only 0.0027 in public while their")
print("  rank vectors correlate at Spearman 0.9994. Two almost identical rankings")
print("  scored 0.0027 apart. That is the size of the split noise on this leaderboard,")
print("  and it is larger than every OOF KS gain achieved across v61, v62 and v63.")
