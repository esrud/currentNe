# currentNe
Estimation of current effective population using artificial neural networks
# Prerequisites
### g++
It should be included in (almost) every linux distribution. To install it in debian-like distributions:
```
sudo apt install g++
```
### make
This is only needed if you want to use the make commands to compile the program. You could directly compile it using g++.To install it just run the following command (in debian-like distributions):
```
sudo apt install make
```
# Installation
Clone the github repo:
```
git clone https://github.com/esrud/currentNe
```
Compile:
```
cd currentNe
make
```
Another way to compile the program, which will link it statically:
```
make static
```
The program has been tested on Arch Linux, Ubuntu 20.04 and Debian Buster, using g++ version 7.2.0 and above as a compiler.

# Usage
Please note that the program needs at least 3 Gb of free RAM to be able to run. This is dependent on the amount of loci and individuals to be sampled, as well as the maximum number of chromosomes and the maximum distance between loci.
This can be increased (or reduced) by tweaking the constants *MAXLOCI*, *MAXIND*, *MAXCROMO* and *MAXDIST* respectively.
Note that increasing those values will increase the free RAM requirements.
```
currentNe - Current Ne estimator (v1.0 - Jan 2023)
Authors: Enrique Santiago - Carlos Köpke

USAGE: ./currentNe [OPTIONS] <file_name_with_extension> <number_of_chromosomes>
         Where file_name is the name of the data file in vcf, ped or tped
             formats (include the corresponding extension vcf, ped or tped
             in the filename).
         If a .map file is present in the same directory, an estimate based
             on loci in diferent chromosomes will also be calculated.

 OPTIONS:
   -h    Print out this manual.
   -s    Number of SNPs to use in the analysis (all by default).
   -k    -If a POSITIVE NUMBER is given, the number of full siblings that
         a random individual has IN THE POPULATION (the population is the
         set of reproducers). With full lifetime monogamy k=2, with 50%
         of monogamy k=1 and so on. With one litter per multiparous
         female k=2, with two litters per female sired by the same father
         k=2 but if sired by different fathers k=1, in general, k=2/Le
         where Le is the effective number of litters (Santiago et al. 2023).
         -If ZERO is specified (i.e., -k 0), each offspring is assumed to
         be from a new random pairing.
         -If a NEGATIVE NUMBER is specified, the average number of full
         siblings observed per individual IN THE SAMPLE. The number k of
         full siblings in the population will be estimated along with Ne.
         -BY DEFAULT, i.e. if the modifier is not used, the average number
         of full siblings k will be estimated from the input data.
   -o    Specifies the output filename. If not specified, the output
         filename is built from the name of the input file.
   -t    Number of threads (default: 8)
   -q    Run quietly. Only prints out Ne estimation
   -p    Print the analysis to stdout. If not specified a file will be created
   -v    Only used with -q. Prints also the bounds for the two confidence
         intervals of 50% and 90%.

EXAMPLES:
   - Random mating and 20 chromosomes (equivalent to a genome of 20 Morgans),
     assuming that full siblings are no more frequent than expected  under
     random pairing (each offspring from a new random pairing).
         ./currentNe -k 0 filename 20
   - Same as before but only a random subsample of 10000 SNPs
     will be analysed:
         ./currentNe -k 0 -s 100000 filename 20
   - Same as before but full siblings could be more frequent than expected
     under random pairing. Full siblings will be identified from the
     genotyping data in the ped file:
         ./currentNe -s 100000 filename 20
   - Two full siblings per individual (k = 2) IN THE POPULATION:
         ./currentNe -k 2 filename 20
   - An 80% of lifetime monogamy in the population. Output filename specified:
         ./currentNe -k 1.6 -o SS81out filename 20
     (with a monogamy rate m = 0.80, the expected number of full
     siblings that a random individual has is k = 2*m = 1.6)
   - If 0.2 full siblings per individual are OBSERVED IN THE SAMPLE:
         ./currentNe -k -0.2 filename 20
     (NOTE the MINUS SIGN before the number of full sibling 0.2)
```
