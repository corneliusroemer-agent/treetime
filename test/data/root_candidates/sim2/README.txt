Simulated dataset sim2
leaves N=14, columns L=400, clock rate=0.0012 substitutions/column/time unit
tree seed=201, alignment seed=1201 (numpy default_rng for tree, SeqGen rng_seed for columns), model JC69
true root time 1993.23 (16.77 time units before earliest leaf at 2010.0); leaf dates 2010.0-2020.0
true_tree.nwk: rooted generating tree (reference only; branch length = rate * time span)
iqtree command (run in a temp dir): /opt/conda/envs/bioinfo/bin/iqtree --pathogen-force -s aln.fasta -m JC --prefix fit -T 1 -seed 1 -redo
collapse threshold: internal non-root branches with length < 0.5/L = 0.00125; all such branches collapsed into their parent (no maximum degree)
internal branches collapsed: 9
max node degree after collapse (unrooted view): 8
true root edge (split of the two sides of the true root) present in fitted.nwk: yes
side of size 7 (t01,t05,t09,t10,t12,t13,t14): trivial=False, split present in fitted.nwk=yes
side of size 7 (t02,t03,t04,t06,t07,t08,t11): trivial=False, split present in fitted.nwk=yes
