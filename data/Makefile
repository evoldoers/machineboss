all: prot1sub.json

HISTORIAN_DIR = ../../historian
prot1.json:
	$(HISTORIAN_DIR)/bin/testmatexp $(HISTORIAN_DIR)/model/prot1.json 1 >$@

prot1sub.json: prot1.json
	node -e '[model=JSON.parse(fs.readFileSync("$<").toString());result={};alph=Object.keys(model.insVec);alph.forEach((c)=>{result["eqm"+c.toUpperCase()]=model.insVec[c]});alph.forEach((c)=>{alph.forEach((d)=>{result["sub"+c.toUpperCase()+d.toUpperCase()]=model.subMat[c][d]})});console.log(JSON.stringify(result))' >$@

Ecoli_codon.html:
	curl 'http://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=83333&aa=1&style=N' >$@

Ecoli_codon.json: Ecoli_codon.html
	node -e 'html=fs.readFileSync("$<").toString();re=/([ACGU]{3,3}) ([A-Z]) ([0-9\. ]{4,4}) ([0-9\. ]{4,4}) \(([ 0-9]{6,6})\)/g;codFreq={};aaFreq={};cod2aa={};cods=[];while((match=re.exec(html))){cod=match[1];aa=match[2];freq=parseFloat(match[4])/1000;codFreq[cod]=freq;aaFreq[aa]=(aaFreq[aa]||0)+freq;cod2aa[cod]=aa;cods.push(cod)}result={};cods.forEach(function(cod){result[cod2aa[cod]+"_"+cod]=codFreq[cod]/aaFreq[cod2aa[cod]]});sorted={};Object.keys(result).sort().forEach(function(p){sorted[p]=result[p]});console.log(JSON.stringify(sorted))' >$@
