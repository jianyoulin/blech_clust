parallel -k -j 31 --noswap --load 100% --progress --memfree 4G --retry-failed --joblog /mnt/f/Experiment_Umami/test_recording/LL2_60min_SNMX_210312_134527/results.log bash blech_clust_jetstream_parallel1.sh ::: {1..32}