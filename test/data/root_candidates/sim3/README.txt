Simulated dataset sim3
leaves N=24, columns L=300, clock rate=0.006 substitutions/column/time unit
tree seed=303, alignment seed=1303 (numpy default_rng for tree, SeqGen rng_seed for columns), model JC69
true root time 1992.76 (17.24 time units before earliest leaf at 2010.0); leaf dates 2010.0-2020.0
true_tree.nwk: rooted generating tree (reference only; branch length = rate * time span)
iqtree command (run in a temp dir): /opt/conda/envs/bioinfo/bin/iqtree --pathogen-force -s aln.fasta -m JC --prefix fit -T 1 -seed 1 -redo
collapse threshold: internal non-root branches with length < 0.5/L = 0.00166667; all such branches collapsed into their parent (no maximum degree)
internal branches collapsed: 4
max node degree after collapse (unrooted view): 4
true root edge (split of the two sides of the true root) present in fitted.nwk: yes
side of size 11 (t06,t07,t10,t11,t12,t15,t17,t18,t19,t20,t22): trivial=False, split present in fitted.nwk=yes
side of size 13 (t01,t02,t03,t04,t05,t08,t09,t13,t14,t16,t21,t23,t24): trivial=False, split present in fitted.nwk=yes
