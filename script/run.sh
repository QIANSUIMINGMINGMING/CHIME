# #!/bin/bash
# #ops
# read=(100 50 95 0 0)
# insert=(0 0 0 100 5)
# update=(0 50 5 0 0)
# delete=(0 0 0 0 0)
# range=(0 0 0 0 95)

# #fixed-op
# readonly=(100 0 0 0 0)
# updateonly=(0 0 100 0 0)

# #exp
# threads=(0 2 18 36 72 108 144)
# #threads=(0 2 16 32 64 96 128)
# mem_threads=(1 4)
# cache=(0 64 128 256 512 1024)
# distribution=(0 1 2 3)
# zipf=(0.99)
# bulk=200
# warmup=10
# runnum=200
# nodenum=5

# #other
# correct=0
# timebase=1
# early=1
# #index=(0 1 2)
# rpc=1
# admit=0.1
# tune=0

# SELF_IP=$(hostname -I | awk '{print $1}')

# for dis in 0
# do 
#     for op in 0
#     do 
#         for idx in 0
#         do  
#             for t in 1
#             do
#                 # ./restartMemc.sh
#                 sudo ../build/newbench $nodenum ${read[$op]} ${insert[$op]} ${update[$op]} ${delete[$op]} ${range[$op]} ${threads[$t]} ${mem_threads[0]} ${cache[3]} ${distribution[$dis]} ${zipf[0]} $bulk $warmup $runnum $correct $timebase $early $idx $rpc $admit $tune 36
#                 sleep 2
#             done
#         done
#     done
# done

# #!/bin/bash
# # ops
# read=(100 50 95 50 0)
# insert=(0 0 0 50 5)
# update=(0 50 5 0 0)
# delete=(0 0 0 0 0)
# range=(0 0 0 0 95)

# # fixed-op
# readonly=(100 0 0 0 0)
# updateonly=(0 0 100 0 0)

# # exp
# # threads=(0 2 18 36 72 108 144)
# threads=(0 4 24 48 96 144 192)
# # threads=(0 2 16 32 64 96 128)
# mem_threads=(1 4)
# cache=(0 64 128 256 384 512 1024)
# distribution=(0 1 2 3)
# zipf=(0.99)
# bulk=60
# warmup=10
# runnum=60
# nodenum=5


# SELF_IP=$(hostname -I | awk '{print $1}')

# # Take four parameters for dis, op, idx, and t
# dis=$1
# op=$2
# idx=$3
# t=$4
# cpupercentage=$5

# # Validate input arguments (optional)
# if [ -z "$dis" ] || [ -z "$op" ] || [ -z "$idx" ] || [ -z "$t" ] || [ -z "$cpupercentage" ] ; then
#     echo "Usage: $0 <dis> <op> <idx> <t> <cpupercentage>"
#     exit 1
# fi

# # Execute the command with provided parameters
# sudo ~/ccpro/CHIME/build/benchmark $nodenum ${read[$op]} ${insert[$op]} ${update[$op]} ${delete[$op]} ${range[$op]} ${threads[$t]} ${mem_threads[0]} ${cache[4]} ${distribution[$dis]} ${zipf[0]} $bulk $warmup $runnum 48 

# sleep 2

# # 21 5 100 0 0 0 0 144 1 256 0 0.99 50 10 50 0 1 1 3 1 0.1 0 36

#!/bin/bash

# Arrays for different parameters
read=(100 50 95 50 0)
insert=(0 0 0 50 5)
update=(0 50 5 0 0)
delete=(0 0 0 0 0)
range=(0 0 0 0 95)

# fixed-op
readonly=(100 0 0 0 0)
updateonly=(0 0 100 0 0)

# exp
threads=(0 4 24 48 96 144 192)
mem_threads=(1 4)
cache=(0 64 128 256 384 512 1024)
distribution=(0 1 2 3)
zipf=(0.99)
bulk=60
warmup=1
runnum=5
nodenum=5

SELF_IP=$(hostname -I | awk '{print $1}')

# Take four parameters for dis, op, idx, and t
dis=$1
op=$2
idx=$3
t=$4
cpupercentage=$5

# Validate input arguments
if [ -z "$dis" ] || [ -z "$op" ] || [ -z "$idx" ] || [ -z "$t" ] || [ -z "$cpupercentage" ] ; then
    echo "Usage: $0 <dis> <op> <idx> <t> <cpupercentage>"
    exit 1
fi

# Execute the benchmark command and capture its exit status
sudo ~/ccpro/CHIME/build/benchmark $nodenum ${read[$op]} ${insert[$op]} ${update[$op]} ${delete[$op]} ${range[$op]} ${threads[$t]} ${mem_threads[0]} ${cache[4]} ${distribution[$dis]} ${zipf[0]} $bulk $warmup $runnum 48
benchmark_exit_status=$?

# Sleep for 2 seconds
sleep 2

# Exit with the same status as the benchmark program
exit $benchmark_exit_status