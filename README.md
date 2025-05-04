# gpu-sorting

adapted from https://github.com/b0nes164/GPUSorting

To build torch extension:
```
python setup.py build_ext --inplace 
```

Performance under 3090:
```
                512       1024      2048      4096      8192      16384     32768     65536     131072    262144    524288    1048576   2500000   4000000   6000000
Size                                                                                                                                                               
torch            0.0326    0.0331    0.0346    0.0401    0.0592    0.0610    0.0629    0.0691    0.0756    0.0959    0.1362    0.2189    0.4503    0.6810    0.9820
onesweep_auto  *0.0318*  *0.0292*  *0.0337*  *0.0357*  *0.0387*  *0.0440*  *0.0438*  *0.0463*  *0.0591*  *0.0582*  *0.0619*  *0.0826*  *0.1602*  *0.2329*  *0.3252*
```

## Optimizations:
- improve radix histogramming via fusion of histogram and scan kernels, and pipeling with shared memory for large sequence lengths
- relieve register pressure for onesweep kernel via reduction of work performed per-block

## Todos

- [ ] Current kernel template does not work for other radix sizes
- [ ] implement and test for int64 sorting

