
%.aln: %.csv
	perl -e '$$header=<>;$$header=~s/\r//;chomp$$header;@hdr=split/,/,$$header;@p=();while(<>){chomp;s/\r//;push@p,[split/,/,$$_]}sub randchar{my$$r=rand();my$$i=0;while($$i+1<@hdr&&($$r-=$$_[$$i])>0){++$$i}$$c=$$hdr[$$i];$$x=length($$c)?$$c:"-";return $$x}for$$i(1..100){print join("",map(randchar(@$$_),@p)),"\n"}' $< >$@
