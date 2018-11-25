
%.aln: %.csv
	perl -e '$$header=<>;chomp$$header;@hdr=split/,/,$$header;@p=();while(<>){chomp;push@p,[split/,/,$$_]}sub randchar{my$$r=rand();my$$i=0;while($$i<@hdr&&($$r-=$$_[$$i])>0){++$$i}return $$hdr[$$i]||"-"}for$$i(1..100){print join("",map(randchar(@$$_),@p)),"\n"}' $< >$@
