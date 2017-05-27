# Long
~/bossmachine/bin/nanomachine -d ~/Dropbox/Projects/MinION/loman-r9/nanopore2_20160729_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_95274_ch238_read274_strand.fast5 -f ~/Dropbox/Projects/MinION/loman-r9/nanopore2_20160729_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_95274_ch238_read274_strand.fastq -v8 -w .1

# basecall long
~/bossmachine/bin/nanomachine -d ~/Dropbox/Projects/MinION/loman-r9/nanopore2_20160729_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_95274_ch238_read274_strand.fast5 -v8 -b


# Short
~/bossmachine/bin/nanomachine -d ~/Dropbox/Projects/MinION/loman-r9/nanopore2_20160728_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_66703_ch178_read123_strand.fast5 -f ~/Dropbox/Projects/MinION/loman-r9/nanopore2_20160728_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_66703_ch178_read123_strand.fastq -v8


# No HDF5 libraries
~/bossmachine/bin/nanomachine -r ~/Dropbox/Projects/MinION/loman-r9/nanopore2_20160728_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_66703_ch178_read123_strand.txt -f ~/Dropbox/Projects/MinION/loman-r9/nanopore2_20160728_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_66703_ch178_read123_strand.fastq -v8



# Basecalling test
~/bossmachine/src/nano/node/events2model.js -f ~/Dropbox/Projects/MinION/loman-r9/nanopore2_20160729_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_95274_ch238_read274_strand.fast5 >long.json
rm LOG ; ( ~/bossmachine/bin/nanomachine -d ~/Dropbox/Projects/MinION/loman-r9/nanopore2_20160729_FNFAB24462_MN17024_sequencing_run_E_coli_K12_1D_R9_SpotOn_2_95274_ch238_read274_strand.fast5 -m long.json -v6 -x -b > long.fa ) | & tee LOG
