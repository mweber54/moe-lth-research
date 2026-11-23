# Existing evidence before Dataset B

Dataset A canonical rows: 1650. Independent source seeds are replication units. All six sparsities appear in paired R100−R0 tables; sparse loss, dense loss and gap are separate. No results change the prespecified A/B/C/D contrasts.

## Recovery endpoint

Validation spacing is 50 attempts on the fixed validation set. Tail improvement is loss at 2000 minus loss at 2500; positive means continued improvement. Adjacent differences and residual SD around a descriptive linear tail trend measure observed variation, not independently sampled validation noise. A flat segment is not proof of convergence.

```csv
num_experts,router_age,model_type,sparsity,n_seeds,mean_tail_improvement,sd_tail_improvement,seeds_improving,mean_last_interval_improvement,mean_tail_residual_sd
8,0,dense,0.0,5,0.11208163376191634,0.01043655234203637,5,0.008710499221329781,0.002778969199476623
8,0,sparse,0.75,5,0.1194623079986612,0.008430175679909552,5,0.011804063914502016,0.0036216433030450837
8,0,sparse,0.95,5,0.09986661975960764,0.005957903527874595,5,0.009085448711049215,0.002346426886190462
8,50,dense,0.0,5,0.10239599930456093,0.010575734099606524,5,0.006569066554625502,0.0038008157424077892
8,50,sparse,0.75,5,0.08927623028875424,0.00493975663266556,5,0.00933173506607603,0.0034296416619153864
8,50,sparse,0.95,5,0.1061993343352281,0.004682798403645118,5,0.0092534832499509,0.0019955229735477845
8,100,dense,0.0,5,0.09240051038487715,0.004363896834766248,5,0.008340197920892268,0.003608366178514078
8,100,sparse,0.75,5,0.08133702899128674,0.004091900690427684,5,0.008892541294876377,0.003097799373857819
8,100,sparse,0.95,5,0.10888543814412946,0.003589252581261952,5,0.009683881650071368,0.0020801203572446774
32,0,dense,0.0,5,0.11967789541134936,0.005484509497979754,5,0.011558460129494153,0.00350090062250476
32,0,sparse,0.75,5,0.11499562810109994,0.00514165721927143,5,0.009558513116799316,0.0029778058308303256
32,0,sparse,0.95,5,0.12026719434682138,0.009375040300821355,5,0.010978863832391017,0.0024189526713692213
32,50,dense,0.0,5,0.07966733971663435,0.011882646702199793,5,0.005829022127155881,0.0038799299988907256
32,50,sparse,0.75,5,0.08103060279848177,0.006839565589057522,5,0.008842486011414907,0.00274711388068421
32,50,sparse,0.95,5,0.11637841168396931,0.015691791669961173,5,0.008739250308047541,0.0026797383377803164
32,100,dense,0.0,5,0.07423777524483563,0.01125228770734574,5,0.007477414295027085,0.0032733955202569436
32,100,sparse,0.75,5,0.07892155950956288,0.012318193443851799,5,0.008382695552309282,0.0031728351148831357
32,100,sparse,0.95,5,0.11746618857407283,0.012060581002188119,5,0.009833098189193068,0.0025267720059368596

```

## Router age × sparsity

See all 30 architecture/sparsity rows in `dataset_a_age_contrasts_summary.csv` and all 150 paired seed rows. Gap change equals sparse loss change minus dense loss change; repeated grid cells are not independent replicates.

## Retained capacity

Counts are actual stored-mask ones, not nominal density estimates. The approximately equal-retained comparison uses a declared 1% relative count tolerance and the same seed and age. Matching retained parameters does not match active top-1 compute, routing, shared parameters, optimization or convergence. Available architecture/sparsity combinations:

```csv
num_experts_a,sparsity_a,num_experts_b,sparsity_b
8,0.7,12,0.8
8,0.7,16,0.85
8,0.7,24,0.9
8,0.8,16,0.9
8,0.8,32,0.95
8,0.85,12,0.9
8,0.85,24,0.95
8,0.9,16,0.95
12,0.7,24,0.85
12,0.8,16,0.85
12,0.8,24,0.9
12,0.9,24,0.95
16,0.7,24,0.8
16,0.7,32,0.85
16,0.8,32,0.9
16,0.85,24,0.9
16,0.9,32,0.95
24,0.8,32,0.85

```

These descriptive comparisons motivate the controlled follow-up but do not establish capacity limits or a router optimum.
