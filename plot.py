import yaml
import matplotlib.pyplot as plt
import numpy as np

data = yaml.load(open("perf.dat"))

benchmarks = {
    key: [] for key in set([bm["run_name"].split("/")[0] for bm in data["benchmarks"]])
}

for key in benchmarks:
    xlist = []
    ylist = []
    elist = []
    for bm in data["benchmarks"]:
        name, n = bm["run_name"].split("/")
        x = int(n)
        if name != key or bm["run_type"] != "aggregate":
            continue
        if bm["aggregate_name"] == "mean":
            xlist.append(x)
            ylist.append(bm["cpu_time"])
        if bm["aggregate_name"] == "stddev":
            elist.append(bm["cpu_time"])
    a = np.array
    benchmarks[key] = a(xlist), a(ylist), a(elist)

plt.figure()
for key, (x, y, e) in benchmarks.items():        
    plt.errorbar(x, y, e, fmt="o-", label=key)
plt.loglog()
plt.legend()
plt.xlabel(r"$N_\mathrm{particles}$")
plt.ylabel("CPU time / ns")

plt.figure()
yn = benchmarks["variant_process_span"][1]
for key, (x, y, e) in benchmarks.items():
    plt.errorbar(x, y/yn, e/yn, fmt="o-", label=key)
plt.semilogx()
plt.ylim(0.8, 1.5)
plt.legend()
plt.xlabel(r"$N_\mathrm{particles}$")
plt.ylabel("CPU time ratio")


plt.show()
