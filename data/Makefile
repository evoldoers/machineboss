all: prot1sub.json

HISTORIAN_DIR = ../../historian
prot1.json:
	$(HISTORIAN_DIR)/bin/testmatexp $(HISTORIAN_DIR)/model/prot1.json 1 >$@

prot1sub.json: prot1.json
	node -e 'model=JSON.parse(fs.readFileSync("$<").toString());result={};alph=Object.keys(model.insVec);alph.forEach((c)=>{result["eqm"+c.toUpperCase()]=model.insVec[c]});alph.forEach((c)=>{alph.forEach((d)=>{result["sub"+c.toUpperCase()+d.toUpperCase()]=model.subMat[c][d]})});console.log(JSON.stringify(result))' >$@

