
import geopandas as gpd
import numpy as np
import pickle
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally, cut_edges
from gingleator import Gingleator

BURST_LEN = 10
NUM_DISTRICTS = 7
ITERS = 500
POP_COL = "TOTPOP"
N_SAMPS = 10
SCORE_FUNCT = None
EPS = 0.045
MIN_POP_COL = "BVAP"
THRESHOLD = 0.5


## Setup graph, updaters, elections, and initial partition

print("Reading in Data/Graph", flush=True)
sc_gdf = gpd.read_file("./shapefiles/SC.geojson")
graph = Graph.from_geodataframe(sc_gdf)


my_updaters = {"population" : Tally(POP_COL, alias="population"),
               "VAP": Tally("VAP"),
               "BVAP": Tally("BVAP"),
               "HVAP": Tally("HVAP"),
               "WVAP": Tally("WVAP"),
               "nWVAP": lambda p: {k: v - p["WVAP"][k] for k,v in p["VAP"].items()},
               "cut_edges": cut_edges}

total_pop = sum([graph.nodes()[n][POP_COL] for n in graph.nodes()])

initial_plan = {}
for node, data in graph.nodes(data=True):
    if "CD" in data:
        initial_plan[node] = data["CD"]

init_partition = Partition(
    graph,
    assignment=initial_plan,
    updaters=my_updaters)


gingles = Gingleator(init_partition, pop_col=POP_COL,
                     threshold=THRESHOLD, score_funct=SCORE_FUNCT, epsilon=EPS,
                     minority_perc_col="{}_perc".format(MIN_POP_COL))

gingles.init_minority_perc_col(MIN_POP_COL, "VAP",
                               "{}_perc".format(MIN_POP_COL))

num_bursts = int(ITERS/BURST_LEN)

print("Starting Short Bursts Runs", flush=True)

for n in range(N_SAMPS):
    sb_obs = gingles.short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                     maximize=True, verbose=False)
    print("\tFinished chain {}".format(n), flush=True)

    print("\tSaving results", flush=True)

    f_out = "short_burst_run_result/threshold_{}/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_{}_{}.npy".format(THRESHOLD, 'SC',
                                                        NUM_DISTRICTS, MIN_POP_COL, EPS,
                                                        ITERS, BURST_LEN, 'score0', n)
    np.save(f_out, sb_obs[1])

    f_out_part = "short_burst_run_result/threshold_{}/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_{}_{}_max_part.p".format(THRESHOLD, 'SC',
                                                        NUM_DISTRICTS, MIN_POP_COL, EPS,
                                                        ITERS, BURST_LEN, 'score0', n)

    max_stats = {"VAP": sb_obs[0][0]["VAP"],
                 "BVAP": sb_obs[0][0]["BVAP"],
                 "WVAP": sb_obs[0][0]["WVAP"],
                 "HVAP": sb_obs[0][0]["HVAP"],}

    with open(f_out_part, "wb") as f_out:
        pickle.dump(max_stats, f_out)